# moodboard
moodboard recommendation repo

![screenshot of a match that matches semantic vibe, rather than exact match to photo](/assets/images/pool_party.png)

## links:
### data
+ [toy clothing dataset 1](https://github.com/alexeygrigorev/clothing-dataset)
+ [amazon fashion dataset](https://data.world/promptcloud/amazon-fashion-products-2020)
+ [fashion datasets](https://data.world/datasets/fashion)
+ [ebay feed sdk)](https://github.com/eBay/FeedSDK-Python)

### models
+ [huggingface's clip](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPModel.forward.returns)
+ [blogpost on fine-tuning clip](https://huggingface.co/blog/fine-tune-clip-rsicd)
+ [loss function for clip](https://github.com/huggingface/transformers/blob/v4.29.1/src/transformers/models/clip/modeling_clip.py#L1151)
+ [openai notebook on interacting with CLIP](https://github.com/openai/CLIP/blob/main/notebooks/Interacting_with_CLIP.ipynb)

+ [LiT: Zero-shot transfer with locked image tuning](https://huggingface.co/docs/transformers/model_doc/vision-text-dual-encoder)
    + another approach said to do well for zero-shot transfer that may perform better than CLIP after fine-tuning

### utils
+ [vector search libraries](https://github.com/currentslab/awesome-vector-search)
    + [faiss](https://github.com/facebookresearch/faiss/wiki/Getting-started): a vector search library for doing nearest neighbors for recommendation
    + [scann](https://github.com/google-research/google-research/blob/master/scann/docs/example.ipynb) more intense vector search with examples

## TODO:
1. run untuned CLIP model over clothing dataset, filtered by clothing category
    + text-to-image rec
    ~~+ image-to-image rec (see ipynb in repo for how to get image features only)~~
    ~~+ run search in scalable manner using vector database for candidates such as [faiss](https://github.com/facebookresearch/faiss/wiki/Getting-started)~~

2. replicate fine-tuning clip
3. find a new clothing or product dataset with textual metadata of the products
    + fine-tune clip mapping product images to sentences created from the metadata

## NOTES:
+ focus on quality of match over 1000s/10,000s of products, rather than lower quality match over millions of products
    + etsy and ebay run over millions, but the match quality is low to the point of not really useful.
+ focus on user interface - adding the extra step where the user says what type of product they are looking for
    + etsy and ebay suffer from trying to exactly match the photo, to the point where they're constantly recommending actual photos.  this one extra step can dramatically improve the quality of our matches, frees up the ability to search on "gist" rather than exact match.  It may also be more fun for the user, as we educate them on searching via gist/vibe rather than exact match.  finally, this is a true differentiator from the other methods.  if we show it's a better experience, this takes us out of direct competition with etsy/ebay current offerings, and gives routes perchance even to patents.
