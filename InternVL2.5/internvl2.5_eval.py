import torch
from transformers import AutoTokenizer, AutoModel
from utils import build_transform, find_closest_aspect_ratio, dynamic_preprocess, load_image
import json
import os
from tqdm import tqdm
from PIL import Image

def eval(args
         ):
    # Process the question file
    with open(args.question_path,'r') as f:
        questions = [json.loads(line) for line in f]

    # questions = questions[506:] # Uncomment to skip first 506 questions
    answers_file = os.path.expanduser(args.answer_path)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "a")
    

    for line in tqdm(questions, total=len(questions)):

        idx = line.get("question_id", line.get("id"))   # Use "question_id" if it exists, otherwise use "id"

        if "object" in line: # For YO-LLaVA tasks
            text = "Is it the same " + line["object"] + " in the first picture and the second picture?"
        else: # For MPOPE tasks
            text = line["text"]


        _question = text
        text += " Please answer with yes or no."
        image_file_list = line["image_list"]
        image_path_list = [os.path.join(args.image_folder,i) for i in image_file_list]
        num_images = line["num_images"]
        label = line["label"]
        text = '<image>\n' + text


        # Multi-image multi-round conversation, combined images 
        image_list = [load_image(i, max_num=12).to(torch.bfloat16).cuda() for i in image_path_list]
        # print("num_image: ",torch.stack(image_list, dim=0).shape)
        pixel_values = torch.cat(image_list, dim=0)
        print("pixel_values.shape: ",pixel_values.shape)
        generation_config = dict(max_new_tokens=1024, do_sample=False)
        question = text
        response, history = model.chat(tokenizer, pixel_values, question, generation_config,
                                    history=None, return_history=True,alpha=args.alpha,base_ratio=args.base_ratio)
        print(f'User: {question}\nAssistant: {response}')
        
        # Write the answer to the output file
        ans_file.write(json.dumps({"question_id": idx,
                                #    "type": _type,
                                   "image_num": num_images,
                                            "question": _question,
                                            "text": response,                            
                                            "label": label,
                                            }) + "\n")
        ans_file.flush()
    ans_file.close()


if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model_path", default="", type=str)
    argparser.add_argument("--image_folder", default="", type=str)
    argparser.add_argument("--answer_path", default="", type=str)
    argparser.add_argument("--question_path", default="", type=str)
    argparser.add_argument("--max-new-tokens", type=int, default=512)
    argparser.add_argument("--load-8bit", action="store_true")
    argparser.add_argument("--load-4bit", action="store_true")
    argparser.add_argument("--alpha", type=float, default=0)
    argparser.add_argument("--base_ratio", type=float, default=0)
    args = argparser.parse_args()

    path = args.model_path
    model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=False,
        trust_remote_code=True
        ).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

    

    eval(args)
