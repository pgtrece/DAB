import os
import json
from tqdm import tqdm
from transformers import AutoModel, BitsAndBytesConfig
import torch
from PIL import Image
from mantis.models.mllava import chat_mllava
from PIL import Image
import torch
from mantis.models.mllava import MLlavaProcessor, LlavaForConditionalGeneration


def eval(question_file,
         answers_file,
         image_folder,
         ):
    # Load the question file
    with open(question_file, 'r') as f:
        questions = [json.loads(line) for line in f]

    # questions = questions[536:]  # Uncomment to process a subset of questions
    answers_file = os.path.expanduser(answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "a")
    

    for line in tqdm(questions, total=len(questions)):

        idx = line.get("question_id", line.get("id"))   # Use "question_id" if exists, otherwise use "id"

        if "object" in line: # For YO-LLaVA tasks
            text = "Is it the same " + line["object"] + " in the first picture and the second picture?"
        else: # For MPOPE tasks
            text = line["text"]

        
        if args.yes_or_no:
            text += " Please answer with yes or no."
        question = text
        image_file_list = line["image_list"]
        image_path_list = [os.path.join(image_folder, i) for i in image_file_list]

        
        if "injected" in line:
            injected = line["injected"]
        else:
            injected = False

        num_images = line["num_images"]
        if "label" in line:
            label = line["label"]
        else:
            label = None

        images_list = []
        for i in range(num_images):
            image_obj = Image.open(image_path_list[i]).convert("RGB")
            image_obj = image_obj.resize((384, 384))    
            images_list.append(image_obj)

        
        generation_kwargs = {
            "max_new_tokens": 128,
            "num_beams": 1,
            "do_sample": False,
            "alpha": args.alpha,
            "base_ratio": args.base_ratio,
        }
        for i in range(num_images):
            text = '<image>' + text
        
        # Debug: Print image type and properties
        # print(type(images_list[0]), getattr(images_list[0], 'shape', None))
        # print(images_list[0].mode, images_list[0].size)
        
        response, _ = chat_mllava(text, images_list, model, processor, **generation_kwargs)

        print("ASSISTANT: ", response)
        
        # Write the answer to the output file
        ans_file.write(json.dumps({"question_id": idx,
                                   "image_num": num_images,
                                   "question": question,
                                   "text": response,                            
                                   "label": label,
                                   "injected": injected,
                                   }) + "\n")
        ans_file.flush()
    ans_file.close()


if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model_path", default="", type=str)
    argparser.add_argument("--image_folder", default="", type=str)
    argparser.add_argument("--answers_path", default="", type=str)
    argparser.add_argument("--question_path", default="", type=str)
    argparser.add_argument("--max-new-tokens", type=int, default=512)
    argparser.add_argument("--load-8bit", action="store_true")
    argparser.add_argument("--load-4bit", action="store_true")
    argparser.add_argument("--yes_or_no", type=int, default=0)
    argparser.add_argument("--alpha", type=float, default=0)
    argparser.add_argument("--base_ratio", type=float, default=0.2)
    args = argparser.parse_args()

    model_path = args.model_path
    # Configure 4-bit quantization
    quant_config = BitsAndBytesConfig(load_in_4bit=True)
    processor = MLlavaProcessor.from_pretrained(model_path)
    attn_implementation = "eager"  # or "flash_attention_2"
    model = LlavaForConditionalGeneration.from_pretrained(
        model_path, 
        device_map="cuda", 
        torch_dtype=torch.bfloat16, 
        attn_implementation=attn_implementation
    )

    eval(
         question_file=args.question_path,
         answers_file=args.answers_path,
         image_folder=args.image_folder,
         )