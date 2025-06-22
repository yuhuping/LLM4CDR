import sys
import fire
import gradio as gr
import torch
torch.set_num_threads(1)
import transformers
import json
import os
from tqdm import tqdm
import numpy as np
import random

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
from peft import PeftModel, LoraConfig, set_peft_model_state_dict
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, AutoTokenizer
from safetensors.torch import load_file


device = "cuda" if torch.cuda.is_available() else "cpu"
try:
    if torch.backends.mps.is_available():
        device = "mps"
except Exception:
    pass

def load_model(base_model: str, lora_weights: str, load_8bit: bool = False):
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )

    # 加载 LoRA 配置
    config = LoraConfig.from_pretrained(lora_weights)
    # 如果有 safetensors 文件，请使用以下代码
    lora_weights_path = os.path.join(lora_weights, "adapter_model.safetensors")
    if os.path.exists(lora_weights_path):
        # 加载权重
        state_dict = load_file(lora_weights_path)
        # 应用权重到模型
        model = set_peft_model_state_dict(model, state_dict)
    else:
        # 如果有 bin 文件，请使用以下代码
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map="auto",
        )
    tokenizer.padding_side = "left"
    # 修正模型配置
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()  # half precision
    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
    return model, tokenizer

def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""

def evaluate_on_dataset(model, tokenizer, dataset_path, batch_size=8):
    """
    在整个数据集上评估模型，并计算 MAE 指标，同时显示进度条，支持批量执行
    """
    # 读取 JSON 文件
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    mae = 0.0
    mse = 0.0  # 用于计算 RMSE
    n_samples = len(dataset)
    # 打乱数据顺序
    random.shuffle(dataset)
    
    # 使用 tqdm 显示进度条
    pbar = tqdm(total=n_samples, desc="Evaluating")
    for i in range(0, n_samples, batch_size):
        batch = dataset[i:i+batch_size]
        # 批量生成预测
        generated_texts, preds = evaluate_batch(model, tokenizer, batch)
        
        # 计算绝对误差和均方误差
        actual_ratings = np.array([float(sample.get("output", 0.0)) for sample in batch])
        preds = np.array(preds)
        absolute_errors = np.abs(actual_ratings - preds)
        mae += np.sum(absolute_errors)
        mse += np.sum((actual_ratings - preds) ** 2)  # 累积均方误差
        
        # 更新进度条
        pbar.update(len(batch))
        
        # 计算当前已处理样本数
        processed_samples = min(i + batch_size, n_samples)
        
        # 计算当前的 MAE 和 RMSE
        current_mae = mae / processed_samples
        current_rmse = np.sqrt(mse / processed_samples)
        
        # 更新进度条描述信息
        pbar.set_description(f"Evaluating (MAE: {current_mae:.4f}, RMSE: {current_rmse:.4f})")
    
    pbar.close()
    
    # 计算平均绝对误差
    mae /= n_samples
    rmse = np.sqrt(mse / n_samples)  # 计算 RMSE
    return mae, rmse

def evaluate_batch(model, tokenizer, batch, temperature=0, top_p=1, top_k=40, num_beams=1, max_new_tokens=8):
    """
    对批量样本生成回答，并返回生成结果及预测值
    """
    prompts = []
    for sample in batch:
        instruction = sample.get("instruction", "")
        input_text = sample.get("input", "")
        prompts.append(generate_prompt(instruction, input_text))
    
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
    )
    with torch.no_grad():
        generation_output = model.generate(
            **inputs,
            do_sample=False,         # 不启用 sampling
            num_beams=1,             # 不使用 beam search
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,  # ← 添加这一行！
        )
    # 第一新生成 token 的分数（scores 是一个列表，每一步对应一个 tensor）
    # 注意：如果生成了多于一个 token，这里仅取第一步
    first_step_scores = generation_output.scores[0]  # shape: (batch_size, vocab_size)
    # 对 scores 做 softmax 得到概率
    probs = torch.softmax(first_step_scores, dim=-1)  # shape: (batch_size, vocab_size)

    # 得到生成的序列
    generated_ids = generation_output.sequences
    generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    # 提取特定 token ID 的概率
    target_token_ids = [16, 17, 18, 19, 20]  # 对应 "1", "2", "3", "4", "5"
    target_probs = probs[:, target_token_ids].cpu().numpy()  # 提取特定 token ID 的概率

    # 归一化处理
    normalized_probs = target_probs / np.sum(target_probs, axis=1, keepdims=True)

    # 将结果与对应的数字关联
    digits = ["1", "2", "3", "4", "5"]
    preds = np.array([np.dot(normalized_prob, np.array([1, 2, 3, 4, 5])) for normalized_prob in normalized_probs])

    return generated_texts, preds


def test_single_sample():
    # 这里的 base_model、lora_weights 请根据你的实际路径进行设定
    base_model = "/root/autodl-tmp/Meta-Llama-3-8B-Instruct"  # 例如
    lora_weights = "./lora-regression/task1"  # 例如
    dataset_path = "./data/task1/Alltest.json"  # 测试数据集路径
    model, tokenizer = load_model(base_model, lora_weights, load_8bit=True)

    # 在数据集上评估
    mae, rmse = evaluate_on_dataset(model, tokenizer, dataset_path)
    print(f"\nMAE on the dataset: {mae:.4f}")
    print(f"RMSE on the dataset: {rmse:.4f}")

    # 测试数据——根据你的描述构造
    sample_data = {
        "instruction": "Given the user's preference in Movies, identify whether the user will like the target music by answering on a scale of 1 to 5.",
    "input": "User Preference: \"Prokofiev: Ivan the Terrible\"、\"Glenn Gould: Hereafter\"、\"Lortzing - Zar und Zimmermann\"、\"Berlioz - Romeo et Juliette / Hanna Schwarz, Philip Langridge, Peter Meven, Colin Davis, Bavarian Radio Symphony Orchestra\"、\"Mozart: Die Zauberfl&ouml;te\"、\"Keeping Score: The Making of a Performance - Tchaikovsky's 4th Symphony\"、\"Rossini - Il Viaggio a Reims / Bayo, Bros, Merced, Rasmussen, Tarver, Cantarero, Dara, Cobos, Barcelona Opera\"、\"Strauss: Capriccio\"、\"Berlioz - Les Troyens / Graham, Antonacci, Kunde, Tezier, Naouri, Pokupic, Gardiner, Chatelet Opera\"、\"Mussorgsky - Boris Godunov / Matti Salminen, Philip Langridge, Eric Halfvarson, Par Lindkog, Albert Schagidullin, Anatoli Kotxerga, Brian Asawa, Barcelona Opera\"、\"Mozart - Don Giovanni / Giorgio Strehler &middot; Riccardo Muti &middot; T. Allen &middot; E. Gruberova &middot; Teatro alla Scala\"、\"Bernstein - Wonderful Town / Audra McDonald, Kim Criswell, Thomas Hampson, Wayne Marshall, Simon Rattle, Berlin Philharmonic\"、\"Richard Strauss - Ariadne auf Naxos / Jurinac, Grist, Hillebrecht, Thomas, Schoffler, Bohm, Salzburg Opera\"、\"Strauss: Der Rosenkavalier\"、\"Gala Concert from St. Petersburg / Anna Netrebko, Dmitri Hvorostovsky, Mischa Maisky, Victor Tretyakov, Elisso Virsaladze, Yuri Temirkanov, Nikolai Alekseev, St. Petersburg Philharmonic\"、\"Bizet - Carmen / Kleiber, Domingo, Obraztsova [VHS]\"、\"Bizet - Carmen / Domashenko, Berti, Aceto, Dashuk, Pastorello, Josipovic, Lombard, Verona Opera\"、\"Weber - Der Freischutz\"、\"Britten - Gloriana / Josephine Barstow, Tom Randle, Emer\"、\"Jacqueline du Pre In Portrait\"、\"Europa Konzert From Lisbon / Pierre Boulez, Maria Joao Pires, Berliner Philharmoniker\"、\"Cilea - Adriana Lecouvreur / Lamberto Puggelli &middot; Gianandrea Gavazzeni &middot; Mirella Freni &middot; Fiorenza Cossotto &middot; Teatro alla Scala\"、\"Paisiello - Nina / Bartoli, Kaufmann, Polgar, Galstian, Veccia, Fischer, Zurich Opera\"、\"Tchaikovsky - Eugene Onegin / Gavrilova, Redkin, Baskov, Novak, Martirosyan, Udalova, Arkhipov, Ermler, Moscow\"、\"Mozart - Le Nozze di Figaro / Te Kanawa, Cotrubas, von Stade, Luxon, Skram, Fryatt; Pritchard, Glyndebourne Opera\"、\"Amilcare Ponchielli: La Gioconda\"、\"Vienna State Opera Gala Concert / Domingo, Terfel, Gruberova, Urmana, Hampson, Baltsa, Kirchschlager, Polaski, Struckmann, Schade, Furlanetto\"、\"Arturo Benedetti Michelangeli - Beethoven Schubert Brahms\"、\"Mozart - Cosi Fan Tutte / Muti, Dessi, Ziegler, Teatro alla Scala\"、\"Bruckner/Beethoven - Symphony No. 7, Piano Concerto No. 3, Alfred Brendel, Claudio Abbado\"\nWhether the user will like the target music \"So Much to Tell\"?Users who interacted with the target music in the source domain also liked: \"Verdi - Aida - The Metropolitan Opera/James Levine [VHS]\"、\"Donizetti - L'Elisir d'Amore / Eschwe, Netrebko, Villazon, Wiener Staatsoper\"、\"Handel - Rodelinda / Antonacci, Scholl, Streit, Chiummo, Winter, Stefanowicz, Christie, Glyndebourne Opera\"、\"Rameau - Les Indes Galantes / Petibon, Croft, Hartelius, Agnew, Rivenq, Berg, Strehl, Christie, Les Arts Florissants, Paris Opera\"、\"Rameau: Platee ~ Agnew, Delunsch, Beuron, Naouri, Le Texier, Lamprecht, Minkowski, Paris Opera\"、\"Balanchine - Jewels / Aurelie Dupont, Alessio Carbone, Marie-Agnes Gillot, Agnes Letestu, Jean-Guillaume Bart, Clairemarie Osta, Kader Belarbi, Paris Opera Ballet\"、\"Richard Strauss - Arabella / Thielemann, Te Kanawa, Brendel, Metropolitan Opera\"、\"Gilbert &amp; Sullivan's The Mikado / English National Opera [VHS]\"、\"Saariaho: L'Amour de loin\"、\"Young Guns\"、\"Nowhere in Africa\"、\"Out of Time\"、\"The Housekeeper\"、\"Russian Ark [VHS]\"、\"Jan Dara\"、\"Million Dollar Baby (Two-Disc Widescreen Edition)\"、\"Italian for Beginners\"、\"I am Sam (New Line Platinum Series)\"、\"Young Adam\"",
    "output": "5.0"
    }
    # prompt = generate_prompt(sample_data)
    # inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    # print(f"Tokenized input length: {inputs['input_ids'].shape[1]}")
    
    
    # print("测试样本：")
    # print(json.dumps(sample_data, indent=2, ensure_ascii=False))
    # print("\n模型生成中...")

    # # 批量测试示例
    # batch_size = 2
    # dataset = [sample_data] * batch_size  # 创建一个包含 batch_size 个相同样本的批量
    
    # generated_texts, preds = evaluate_batch(model, tokenizer, dataset)
    # for i in range(1):
    #     print(f"\n样本 {i+1} 的生成结果：")
    #     print(generated_texts[i])
        
    #     print(f"\n样本 {i+1} 的预测值（加权平均）：")
    #     print(f"{preds[i]:.4f}")


if __name__ == "__main__":
    # 可通过命令行传入参数，这里我们直接调用测试函数
    test_single_sample()