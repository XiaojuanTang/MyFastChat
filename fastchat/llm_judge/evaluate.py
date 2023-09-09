"""Generate answers with local models.

Usage:
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""
import argparse
import json
import os
import random
import time
import datetime

import shortuuid
import torch
from tqdm import tqdm


from fastchat.llm_judge.common import load_questions, temperature_config
from fastchat.model import load_model, get_conversation_template



import logging
def get_logger(filename, verbosity=1, name=None, func = 'w'):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    # # Remove any existing handlers
    # for handler in logger.handlers:
    #     logger.removeHandler(handler)

    # Output to file
    fh = logging.FileHandler(filename, func)
    fh.setLevel(level_dict[verbosity])
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    # logger.removeHandler(fh)

    # # Output to terminal
    # sh = logging.StreamHandler()
    # sh.setFormatter(formatter)
    # logger.addHandler(sh)

    return logger



def run_eval(
    model_path,
    model_id,
    question_file,
    question_begin,
    question_end,
    max_new_token,
    num_choices,
    num_gpus_per_model,
    num_gpus_total,
    max_gpu_memory,
    logger,
    logger2
):
    # questions = load_questions(question_file, question_begin, question_end)
    questions = json.load(open(question_file, 'r'))[question_begin: question_end]
    # questions = process_questions(questions_raw)

    # random shuffle the questions to balance the loading
    # random.shuffle(questions)

    # Split the question file into `num_gpus` files
    assert num_gpus_total % num_gpus_per_model == 0
    use_ray = num_gpus_total // num_gpus_per_model > 1

    if use_ray:
        get_answers_func = ray.remote(num_gpus=num_gpus_per_model)(
            get_model_answers
        ).remote
    else:
        get_answers_func = get_model_answers

    chunk_size = len(questions) // (num_gpus_total // num_gpus_per_model) // 2
    ans_handles = []
    for i in range(0, len(questions), chunk_size):
        ans_handles.append(
            get_answers_func(
                model_path,
                model_id,
                questions[i : i + chunk_size],
                # answer_file,
                max_new_token,
                num_choices,
                num_gpus_per_model,
                max_gpu_memory,
                logger,
            )
        )
    
    if use_ray:
        ans_handles = ray.get(ans_handles)
    
    correct_num = 0
    total_num = 0
    for ans in ans_handles:
        correct_num += ans[0]
        total_num += ans[1]
    
    logger.info('Correct: ' + str(correct_num)) 
    logger.info('Total: ' + str(total_num))
    logger.info('Accuracy: ' + str(correct_num / total_num))
    
    logger2.debug("Accuracy: " + str(correct_num / total_num))
    
@torch.inference_mode()
def get_model_answers(
    model_path,
    model_id,
    questions,
    # answer_file,
    max_new_token,
    num_choices,
    num_gpus_per_model,
    max_gpu_memory,
    logger,
):
    model, tokenizer = load_model(
        model_path,
        device="cuda",
        num_gpus=num_gpus_per_model,
        max_gpu_memory=max_gpu_memory,
        load_8bit=False,
        cpu_offloading=False,
        debug=False,
    )
    model = model.merge_and_unload()
    correct_num = 0
    total_num = 0
    for sample in tqdm(questions):
        # if question["category"] in temperature_config:
        #     temperature = temperature_config[question["category"]]
        # else:
        #     temperature = 0.7
        temperature = 0.0
        
        # # question = 'Given a set of rules and facts, you have to reason whether a statement is true or false. Here are some facts and rules: \n' + sample["facts"] + sample['rules'] + "Does it imply that the statement \"" + sample['question'] + '\" is True?'
        # question = 'Given a set of rules and facts, you have to reason whether a statement is true or false. Here are some facts and rules: \n' + '\n'.join(sample['premises']) + "\nDoes it imply that the statement \"" + sample['conclusion'] + '\" is True?'

        
        conv = get_conversation_template(model_id)

        # conv.set_system_message("You are a helpful, respectful and honest assistant with logical reasoning abilities. Given a set of rules and facts, you have to reason whether a statement is true or false.")
        
        conv.append_message(conv.roles[0],sample['conversations'][0]['value']) 
        # conv.append_message(conv.roles[0],sample['conversations'][0]['value'] + " Let's think step by step. ") 

        conv.append_message(conv.roles[1], None)
        
        prompt = conv.get_prompt()

        input_ids = tokenizer([prompt]).input_ids

        if temperature < 1e-4:
            do_sample = False
        else:
            do_sample = True


        # some models may error out when generating long outputs
        try:
            output_ids = model.generate(
                torch.as_tensor(input_ids).cuda(),
                do_sample=do_sample,
                temperature=temperature,
                max_new_tokens=max_new_token,
            )
            if model.config.is_encoder_decoder:
                output_ids = output_ids[0]
            else:
                output_ids = output_ids[0][len(input_ids[0]) :]
            output = tokenizer.decode(
                output_ids,
                skip_special_tokens=True,
                spaces_between_special_tokens=False,
            )
            if conv.stop_str:
                output = output[: output.find(conv.stop_str)]
            output = output.strip()

            if conv.name == "xgen" and output.startswith("Assistant:"):
                output = output.replace("Assistant:", "", 1).strip()
        except RuntimeError as e:
            print("ERROR question ID: ", sample["id"])
            output = "ERROR"
            print(e)

        logger.info(prompt)
        logger.info('prediction: ' + output) 
        logger.info('grounding truth: ' + sample["conversations"][1]['value'])
        total_num += 1
        if output == sample["conversations"][1]['value']:
            logger.info('Correct')
            correct_num += 1
        else:
            logger.info('Wrong')

    # logger.info('Correct: ' + str(correct_num))
    # logger.info('Total: ' + str(total_num))
    # logger.info('Accuracy: ' + str(correct_num / total_num))     
    return correct_num, total_num

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        default="/home/txx/ft_models/llama2-13-chat-folio",
        # default="/home/txx/huggingface/vicuna-7b-v1.5-16k",

        # required=True,
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument("--model-id", type=str, default="vicuna")
    # parser.add_argument("--model-id", type=str, default="Llama-2-7b-hf")
    
    parser.add_argument(
        "--bench-name",
        type=str,
        # default="proofwriter_OWA_symbolic",
        default="/home/txx/datasets/FOLIO/data/train_ft.json",

        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--output_log",
        type=str,
        default="test",
        help="The output log file.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end", type=int, help="A debug option. The end index of questions."
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument(
        "--max-new-token",
        type=int,
        default=512,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1, help="The total number of GPUs."
    )
    parser.add_argument(
        "--max-gpu-memory",
        type=str,
        help="Maxmum GPU memory used for model weights per GPU.",
    )
    
    args = parser.parse_args()
    nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    model_name = args.model_path.split('/')[-1]
    dir = model_name + '_logs/' + str(args.output_log)
    if not os.path.exists(dir):
        os.makedirs(dir)
    logger = get_logger(os.path.join(dir, nowTime + '.log'), verbosity=1, name='logger')

    logger2 = get_logger(os.path.join(model_name + '_logs/', 'results.log'), verbosity=0, name='logger2', func='a')
    logger2.debug(" evaluate model: " +  args.model_path)
    logger2.debug(" data_path: " + args.bench_name)


    if args.num_gpus_total // args.num_gpus_per_model > 1:
        import ray

        ray.init()

    # question_file = f"/home/txx/datasets/{args.bench_name}/depth-{args.depth}/meta-dev_symbolic.jsonl"
    # question_file = f"/home/txx/datasets/{args.bench_name}/data/v0.0/folio-validation.jsonl"
    question_file = args.bench_name
    # if args.answer_file:
    #     answer_file = args.answer_file
    # else:
    #     answer_file = f"/home/txj/projects/LLMs_PG/FastChat/fastchat/llm_judge/data/{args.bench_name}/model_answer/{args.model_id}.jsonl"

    # print(f"Output to {answer_file}")

    run_eval(
        args.model_path,
        args.model_id,
        question_file,
        args.question_begin,
        args.question_end,
        # answer_file,
        args.max_new_token,
        args.num_choices,
        args.num_gpus_per_model,
        args.num_gpus_total,
        args.max_gpu_memory,
        logger,
        logger2
    )
    
    # reorg_answer_file(answer_file)
