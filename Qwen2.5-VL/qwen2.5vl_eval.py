from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import os
import json
from tqdm import tqdm
from transformers import AutoModel, BitsAndBytesConfig
import torch
from PIL import Image

def deal_question(p):
    return "Is there a(an) " + p + " in both images?"


def eval(question_file,
         answers_file,
         image_folder,
         ):

    with open(question_file,'r') as f:
        questions = [json.loads(line) for line in f]

    # questions = questions[536:]  
    answers_file = os.path.expanduser(answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "a")
    

    for line in tqdm(questions, total=len(questions)):

        idx = line.get("question_id", line.get("id"))  


        text = line["text"]
        
        question = text
        if args.yes_or_no:
            text += " Please answer with Yes or No."
        image_file_list = line["image_list"]
        image_path_list = [os.path.join(image_folder,i) for i in image_file_list]
        if "injected" in line:
            injected = line["injected"]
        else:
            injected = None


        num_images = line["num_images"]
        label = line["label"]
        content = []
        for i in range(num_images):
            
            image_obj = Image.open(image_path_list[i])
            image_obj = image_obj.resize((24*28, 24*28))    
            content.append({"type": "image", "image": image_obj})

        content.append({"type": "text", "text": text})

        messages = [
            {
                "role": "user",
                "content": content,
            }
        ]


        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        ###OURS###
        num_image_token = (inputs['input_ids'][0] == 151655).sum().item()
        img_str_idx = (inputs['input_ids'][0] == 151652).nonzero(as_tuple=True)[0].tolist()  # Calculate the position of img_str in the index
        per_img_token = num_image_token/num_images
        img_str_idx.append(per_img_token)  # Add the number of image tokens per image at the end of the array
        if args.alpha == 0:
            img_str_idx = None
        ###OURS###
        inputs = inputs.to("cuda")
        
        # Inference  # img_str_idx = None means baseline
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=128,
            img_str_idx=img_str_idx,
            alpha=args.alpha,
            base_ratio=args.base_ratio,
            )


        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        print(output_text)
        

        ans_file.write(json.dumps({"question_id": idx,
                                   "image_num": num_images,
                                            "question": question,
                                            "text": output_text[0],                            
                                            "label": label,
                                            "injected":injected,
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
    argparser.add_argument("--yes_or_no", type=int, default=0)
    argparser.add_argument("--alpha", type=float, default=0)
    argparser.add_argument("--base_ratio", type=float, default=0.2)
    args = argparser.parse_args()

    model_path = args.model_path

    quant_config = BitsAndBytesConfig(load_in_4bit=True)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, 
        torch_dtype="auto", 
        device_map="auto",
        quantization_config=quant_config,
        attn_implementation="eager",
    )

    min_pixels = 256*28*28  
    max_pixels = 1280*28*28
    processor = AutoProcessor.from_pretrained(model_path, min_pixels=min_pixels, max_pixels=max_pixels)

    eval(
         question_file=args.question_path,
         answers_file=args.answer_path,
         image_folder=args.image_folder,
         )

