from PIL import Image
import requests
import torch
from huggingface_hub import hf_hub_download
import pandas as pd
from open_flamingo import create_model_and_transforms

model, image_processor, tokenizer = create_model_and_transforms(
    clip_vision_encoder_path="ViT-L-14",
    clip_vision_encoder_pretrained="openai",
    lang_encoder_path="anas-awadalla/mpt-1b-redpajama-200b",
    tokenizer_path="anas-awadalla/mpt-1b-redpajama-200b",
    cross_attn_every_n_layers=1
)

import os
import numpy as np
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from pycocoevalcap.cider import cider
from pycocoevalcap.spice import spice
from bert_score import score
import pandas as pd
import numpy as np

checkpoint_path = hf_hub_download("openflamingo/OpenFlamingo-3B-vitl-mpt1b", "checkpoint.pt")
model.load_state_dict(torch.load(checkpoint_path), strict=False)

def get_result(image_path, question):
    
    

    """
    Step 1: Load images
    """
    demo_image_one = Image.open(
        requests.get(
            "http://images.cocodataset.org/val2017/000000039769.jpg", stream=True
        ).raw
    )

    demo_image_two = Image.open(
        requests.get(
            "http://images.cocodataset.org/test-stuff2017/000000028137.jpg",
            stream=True
        ).raw
    )

    query_image = Image.open(image_path)


    """
    Step 2: Preprocessing images
    Details: For OpenFlamingo, we expect the image to be a torch tensor of shape 
     batch_size x num_media x num_frames x channels x height x width. 
     In this case batch_size = 1, num_media = 3, num_frames = 1,
     channels = 3, height = 224, width = 224.
    """
    vision_x = [image_processor(demo_image_one).unsqueeze(0), image_processor(demo_image_two).unsqueeze(0), image_processor(query_image).unsqueeze(0)]
    vision_x = torch.cat(vision_x, dim=0)
    vision_x = vision_x.unsqueeze(1).unsqueeze(0)

    """
    Step 3: Preprocessing text
    Details: In the text we expect an <image> special token to indicate where an image is.
     We also expect an <|endofchunk|> special token to indicate the end of the text 
     portion associated with an image.
    """
    tokenizer.padding_side = "left" # For generation padding tokens should be on the left
    lang_x = tokenizer(
        ["<image>An image of two cats.<|endofchunk|><image>An image of a bathroom sink.<|endofchunk|><image>%s"%question],
        return_tensors="pt",
    )


    """
    Step 4: Generate text
    """
    generated_text = model.generate(
        vision_x=vision_x,
        lang_x=lang_x["input_ids"],
        attention_mask=lang_x["attention_mask"],
        max_new_tokens=20,
        num_beams=3,
    )

    #print("Generated text: ", tokenizer.decode(generated_text[0]))
    
    answer = tokenizer.decode(generated_text[0])[98:-15]
    
    return answer


def split_words(sentence):
    return sentence.split(' ')

def Rouge(GT_caption, generated_caption): # returns list of all rouge_n needed in (precision,recall,fmeasure) for each
    res = []
    scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)
    output = scorer.score(GT_caption, generated_caption)
    for r in output:
        score = output[r]
        res.append((score.precision, score.recall, score.fmeasure))
    return res[0][2], res[1][2], res[2][2]

def Bleu_4(GT_caption, generated_caption):
    GT_words = split_words(GT_caption)
    gen_words = split_words(generated_caption)
    weights = (1./4., 1./4., 1./4., 1./4.)
    return corpus_bleu([[GT_words]], [gen_words], weights)

def METEOR(GT_caption, generated_caption):
    GT_words = split_words(GT_caption)
    gen_words = split_words(generated_caption)
    return meteor_score([GT_words], gen_words)

def CIDEr(GT_caption, generated_caption):
    scorer = cider.Cider()
    return scorer.compute_score({0:[GT_caption]}, {0:[generated_caption]})[0]

def SPICE(GT_caption, generated_caption):
    scorer = spice.Spice()
    return scorer.compute_score({0:[GT_caption]}, {0:[generated_caption]})[0]

def BertScore(GT_caption, generated_caption):
    P, R, F1 = score([generated_caption], [GT_caption], lang="en", verbose=True)
    #return P.tolist()[0], R.tolist()[0], F1.tolist()[0]
    return F1.tolist()[0]

def acc_full(GT_caption, generated_caption):
    GT_caption = GT_caption.lower()
    generated_caption = generated_caption.lower()
    return int(GT_caption in generated_caption)

def acc_part(GT_caption, generated_caption):
    GT_words = split_words(GT_caption.lower())
    generated_caption = generated_caption.lower()
    
    in_generated = 0
    for word in GT_words:
        if word in generated_caption:
            in_generated += 1
            
    return in_generated / len(GT_words)


def answer_evaluation(GT_caption, generated_caption):
    rouge1, rouge2, rougeL = Rouge(GT_caption, generated_caption)
    bleu4 = Bleu_4(GT_caption, generated_caption)
    meteor = METEOR(GT_caption, generated_caption)
    cider= CIDEr(GT_caption, generated_caption)
    spice = SPICE(GT_caption, generated_caption)
    bertscore = BertScore(GT_caption, generated_caption)
    
    return rouge1, rouge2, rougeL, bleu4, meteor, cider, spice, bertscore

old_tsv = pd.read_csv("landmark_all.tsv", sep='\t', encoding='utf-8')

df_results = pd.DataFrame(columns=['GT_answer', 'generated_answer','rouge1', 'rouge2', 'rougeL', 'bleu4', 'meteor', 'cider', 'spice', 'bertscore'])

for i in range(0,3501,50):
    
    print(i)
    GT_answer = old_tsv.loc[i,'answer']
    generated_answer = get_result(old_tsv.loc[i,'image_path'], old_tsv.loc[i,'question'])
    rouge1_res, rouge2_res, rougeL_res, bleu4_res, meteor_res, cider_res, spice_res, bertscore_res = answer_evaluation(GT_answer, generated_answer)
    
    df_results = df_results.append({'GT_answer':GT_answer, 'generated_answer': generated_answer, 'rouge1':rouge1_res, 'rouge2':rouge2_res,'rougeL':rougeL_res, \
                                        'bleu4':bleu4_res, 'meteor':meteor_res, \
                                        'cider':cider_res, 'spice':spice_res, 'bertscore':bertscore_res}, ignore_index=True)
    
df_results.to_csv('openflamingo_landmark.tsv', sep='\t', encoding='utf-8', index= False)