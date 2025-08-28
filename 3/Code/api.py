from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModelForCausalLM
import uvicorn
import json
import datetime
import torch

# 设置设备参数
DEVICE = "cuda"
DEVICE_ID = "0"
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE

# 创建FastAPI应用
app = FastAPI()

# 处理POST请求的端点
@app.post("/")
async def create_item(request: Request):
    global model, tokenizer
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    prompt = json_post_list.get('prompt')
    history = json_post_list.get('history')
    max_length = json_post_list.get('max_length')
    top_p = json_post_list.get('top_p')
    temperature = json_post_list.get('temperature')
    
    # 调用模型进行对话生成
    response, history = model.chat(
        tokenizer,
        prompt,
        history=history,
        max_length=max_length if max_length else 2048,
        top_p=top_p if top_p else 0.7,
        temperature=temperature if temperature else 0.95
    )
    
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    
    # 构建响应JSON
    answer = {
        "response": response,
        "history": history,
        "status": 200,
        "time": time
    }
    
    # 构建日志信息
    log = "[" + time + "] " + '", prompt:"' + prompt + '", response:"' + repr(response) + '"'
    print(log)
    return answer

# 主函数入口
if __name__ == '__main__':
    # 加载预训练的分词器和模型 - PyTorch 2.0优化版本
    tokenizer = AutoTokenizer.from_pretrained(
        "/root/autodl-tmp/ZhipuAI/chatglm3-6b", 
        trust_remote_code=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        "/root/autodl-tmp/ZhipuAI/chatglm3-6b", 
        trust_remote_code=True,
        torch_dtype=torch.float16,  # PyTorch 2.0对float16支持更好
        device_map="auto"  # 利用PyTorch 2.0的自动设备映射
    )
    
    model.eval()  # 设置模型为评估模式
    
    # 启动FastAPI应用
    uvicorn.run(app, host='0.0.0.0', port=6006, workers=1)