# moodboard
moodboard recommendation repo

## links:
+ [toy clothing dataset](https://github.com/alexeygrigorev/clothing-dataset)
+ [huggingface's clip](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPModel.forward.returns)
+ [blogpost on fine-tuning clip](https://huggingface.co/blog/fine-tune-clip-rsicd)
+ [loss function for clip](https://github.com/huggingface/transformers/blob/v4.29.1/src/transformers/models/clip/modeling_clip.py#L1151)
+ [openai notebook on interacting with CLIP](https://github.com/openai/CLIP/blob/main/notebooks/Interacting_with_CLIP.ipynb)

+ [LiT: Zero-shot transfer with locked image tuning](https://huggingface.co/docs/transformers/model_doc/vision-text-dual-encoder)
    + another approach said to do well for zero-shot transfer that may perform better than CLIP after fine-tuning

## TODO:
1. run untuned CLIP model over clothing dataset, filtered by clothing category
    + text-to-image rec
    + image-to-image rec (see ipynb in repo for how to get image features only) 

2. replicate fine-tuning clip
3. find a new clothing or product dataset with textual metadata of the products
    + fine-tune clip mapping product images to sentences created from the metadata
