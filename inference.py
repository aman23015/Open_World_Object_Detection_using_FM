# import os
# import torch
# import argparse
# import numpy as np
# import pandas as pd
# from PIL import Image
# from tqdm import tqdm
# from pathlib import Path
# from torchvision import transforms
# from sklearn.metrics.pairwise import cosine_similarity
# import types

# from FOMO import FOMO, PostProcess, build

# # ---- Argument Parser for GPU ----
# parser = argparse.ArgumentParser()
# parser.add_argument('--gpu', type=int, default=0, help='GPU id to use, e.g., 0 or 1')
# args_cli = parser.parse_args()

# # ---- Setup FOMO args ----
# args = types.SimpleNamespace()
# args.model_name = 'google/owlvit-base-patch16'
# args.dataset = 'Aerial'
# args.image_resize = 768
# args.device = 'cuda:1'  # This will be overwritten by the --gpu argument in main()
# args.post_process_method = 'regular'
# args.num_few_shot = 100
# args.use_attributes = True
# args.att_selection = True
# args.att_adapt = True
# args.att_refinement = True
# args.num_att_per_class = 25
# args.classnames_file = 'known_classnames.txt'
# args.prev_classnames_file = 'known_classnames.txt'
# args.templates_file = 'best_templates.txt'
# args.attributes_file = 'attributes.json'
# args.pred_per_im = 100
# args.image_conditioned = True
# args.image_conditioned_file = 'few_shot_data.json'
# args.unk_methods = 'sigmoid-max-mcm'
# args.unk_method = 'sigmoid-max-mcm'
# args.unk_proposal = False
# args.output_dir = 'tmp/rwod'
# args.prev_output_file = ''
# args.output_file = ''
# args.TCP = '295499'
# args.PREV_INTRODUCED_CLS = 0
# args.CUR_INTRODUCED_CLS = 10
# args.seed = 42
# args.eval = False
# args.viz = False
# args.num_workers = 2
# args.test_set = 'test.txt'
# args.train_set = 'train.txt'
# args.data_root = './data'  # adjust if needed
# args.data_task = 'RWD'
# args.unknown_classnames_file = ''
# args.batch_size = 10

# def load_model_for_inference(model_path, learned_path, args):
#     model, post = build(args)
#     model.load_state_dict(torch.load(model_path, map_location=args.device))
#     learned = torch.load(learned_path, map_location=args.device)
#     model.att_W = learned["att_W"]
#     model.att_embeds = learned["att_embeds"]
#     model.unk_head.att_W = learned["att_W"]
#     model.eval()
#     return model, post

# # ---- Image transform ----
# transform = transforms.Compose([
#     transforms.Resize((args.image_resize, args.image_resize)),
#     transforms.ToTensor()
# ])

# def load_and_embed_images(model, image_dir):
#     embeddings = []
#     ids = []
#     for fname in tqdm(os.listdir(image_dir), desc=f'Embedding {os.path.basename(image_dir)}'):
#         if not fname.lower().endswith(('jpg', 'jpeg', 'png')):
#             continue
#         path = os.path.join(image_dir, fname)
#         try:
#             image = transform(Image.open(path).convert("RGB"))
#         except Exception as e:
#             print(f"❌ Failed to open {path}: {e}")
#             continue
#         image = image.unsqueeze(0).to(args.device)
#         with torch.no_grad():
#             embed = model.image_guided_forward(image)
#         if embed is None or embed[0] is None:
#             continue
#         embeddings.append(embed[0].squeeze().cpu().numpy())
#         ids.append(os.path.splitext(fname)[0])
#     return np.array(embeddings), ids

# def match_and_generate_predictions(model, db_embeddings, db_ids, query_embeddings, query_ids, species_prefix, threshold=0.75):
#     predictions = []
#     for i, query_emb in enumerate(query_embeddings):
#         sims = cosine_similarity([query_emb], db_embeddings)[0]
#         max_sim = sims.max()
#         if max_sim > threshold:
#             best_match = db_ids[sims.argmax()]
#             predictions.append(best_match)
#         else:
#             predictions.append("new_individual")
#     return pd.DataFrame({
#         "image_id": query_ids,
#         "individual_id": predictions
#     })

# def main():
#     model_path = 'tmp/rwod/best_model_Aerial_google_owlvit-base-patch16.pth'
#     learned_path = 'tmp/rwod/fomo_learned_Aerial.pth'
#     model, _ = load_model_for_inference(model_path, learned_path, args)

#     root = '/home/aaditya23006/AMAN/SML/animal-clef-2025'
#     species_folders = {
#         'LynxID2025': 'submission_lynx.csv',
#         'SalamanderID2025': 'submission_salamander.csv',
#         'SeaTurtleID2022': 'submission_seaturtle.csv'
#     }

#     all_dfs = []
#     for species, out_file in species_folders.items():
#         db_dir = os.path.join(root, species, 'database')
#         query_dir = os.path.join(root, species, 'query')

#         db_embeddings, db_ids = load_and_embed_images(model, db_dir)
#         query_embeddings, query_ids = load_and_embed_images(model, query_dir)

#         df = match_and_generate_predictions(model, db_embeddings, db_ids, query_embeddings, query_ids, species)
#         df.to_csv(out_file, index=False)
#         all_dfs.append(df)
#         print(f" Saved predictions to: {out_file}")

#     submission = pd.concat(all_dfs, ignore_index=True)
#     submission.to_csv("final_submission.csv", index=False)
#     print(" Final submission saved as final_submission.csv")

# if __name__ == '__main__':
# #     main()
import os
import torch
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity
import types
import pickle

# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context

from FOMO import FOMO, PostProcess, build

# ---- Argument Parser for GPU ----
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU id to use, e.g., 0 or 1')
args_cli = parser.parse_args()

# ---- Setup FOMO args ----
args = types.SimpleNamespace()
args.model_name = 'google/owlvit-base-patch16'
args.dataset = 'Aerial'
args.image_resize = 768
args.device = f'cuda:{args_cli.gpu}'
args.post_process_method = 'regular'
args.num_few_shot = 100
args.use_attributes = True
args.att_selection = False
args.att_adapt = False
args.att_refinement = False
args.num_att_per_class = 25
args.classnames_file = 'known_classnames.txt'
args.prev_classnames_file = 'known_classnames.txt'
args.templates_file = 'best_templates.txt'
args.attributes_file = 'attributes.json'
args.pred_per_im = 100
args.image_conditioned = True
args.image_conditioned_file = 'few_shot_data.json'
args.unk_methods = 'sigmoid-max-mcm'
args.unk_method = 'sigmoid-max-mcm'
args.unk_proposal = False
args.output_dir = 'tmp/rwod'
args.prev_output_file = ''
args.output_file = ''
args.TCP = '295499'
args.PREV_INTRODUCED_CLS = 0
args.CUR_INTRODUCED_CLS = 10
args.seed = 42
args.eval = False
args.viz = False
args.num_workers = 2
args.test_set = 'test.txt'
args.train_set = 'train.txt'
args.data_root = './data'
args.data_task = 'RWD'
args.unknown_classnames_file = ''
args.batch_size = 48
args.neg_sup_ep = 10       # safe dummy value (won't be used in inference)
args.neg_sup_lr = 1e-5    # safe dummy learning rate

def load_model_for_inference(model_path, learned_path, args):
    print(f" Loading model on {args.device}...")
    model, post = build(args)
    model.load_state_dict(torch.load(model_path, map_location=args.device))
    learned = torch.load(learned_path, map_location=args.device)
    model.att_W = learned["att_W"]
    model.att_embeds = learned["att_embeds"]
    model.unk_head.att_W = learned["att_W"]
    model.eval()
    print(f" Model loaded and set to eval mode.")
    return model, post

# ---- Image transform ----
transform = transforms.Compose([
    transforms.Resize((args.image_resize, args.image_resize)),
    transforms.ToTensor()
])

# def load_and_embed_images(model, image_dir):
#     embeddings = []
#     ids = []
#     print(f" Embedding images from: {image_dir}")
#     for fname in tqdm(os.listdir(image_dir), desc=f'Embedding {os.path.basename(image_dir)}'):
#         if not fname.lower().endswith(('jpg', 'jpeg', 'png')):
#             continue
#         path = os.path.join(image_dir, fname)
#         try:
#             image = transform(Image.open(path).convert("RGB"))
#         except Exception as e:
#             print(f" Failed to open {path}: {e}")
#             continue
#         image = image.unsqueeze(0).to(args.device)
#         with torch.no_grad():
#             embed = model.image_guided_forward(image)
#         if embed is None or embed[0] is None:
#             continue
#         embeddings.append(embed[0].squeeze().cpu().numpy())
#         ids.append(os.path.splitext(fname)[0])
#     print(f" Finished embedding {len(ids)} images.")
#     return np.array(embeddings), ids
def load_and_embed_images(model, image_dir):
    embeddings = []
    ids = []
    print(f"📁 Embedding images from: {image_dir}")
    for fname in tqdm(os.listdir(image_dir), desc=f'Embedding {os.path.basename(image_dir)}'):
        if not fname.lower().endswith(('jpg', 'jpeg', 'png')):
            continue
        path = os.path.join(image_dir, fname)
        try:
            image = transform(Image.open(path).convert("RGB"))
        except Exception as e:
            print(f"❌ Failed to open {path}: {e}")
            continue
        image = image.unsqueeze(0).to(args.device)
        with torch.no_grad():
            embed = model.extract_global_features(image)  # <---- Use CLS token
        embeddings.append(embed.squeeze().cpu().numpy())
        ids.append(os.path.splitext(fname)[0])
    return np.array(embeddings), ids

# def match_and_generate_predictions(model, db_embeddings, db_ids, query_embeddings, query_ids, species_prefix, threshold=0.75):
#     predictions = []
#     print(f" Matching query images against database using cosine similarity...")
#     for i, query_emb in enumerate(tqdm(query_embeddings, desc='Matching queries')):
#         sims = cosine_similarity([query_emb], db_embeddings)[0]
#         max_sim = sims.max()
#         if max_sim > threshold:
#             best_match = db_ids[sims.argmax()]
#             predictions.append(best_match)
#         else:
#             predictions.append("new_individual")
#     print(f" Matching completed for {len(query_ids)} query images.")
#     return pd.DataFrame({
#         "image_id": query_ids,
#         "individual_id": predictions
#     })


# def match_and_generate_predictions_with_metadata(model, metadata_path, species_root, species_prefix, threshold=0.75):
#     metadata = pd.read_csv(metadata_path)
#     species_metadata = metadata[metadata['dataset'] == species_prefix]

#     # Create DB embeddings grouped by identity
#     db_metadata = species_metadata[species_metadata['split'] == 'database']
#     db_embeddings = {}
#     print(f"📁 Building database embeddings for {species_prefix}...")

#     for identity in tqdm(db_metadata['identity'].unique(), desc="DB identities"):
#         identity_images = db_metadata[db_metadata['identity'] == identity]
#         embeds = []
#         for _, row in identity_images.iterrows():
#             img_path = os.path.join(species_root, row['path'])
#             if not os.path.exists(img_path):
#                 continue
#             try:
#                 image = transform(Image.open(img_path).convert("RGB"))
#                 image = image.unsqueeze(0).to(args.device)
#                 with torch.no_grad():
#                     embed = model.extract_global_features(image)
#                 if embed is not None and embed[0] is not None:
#                     embeds.append(embed[0].squeeze().cpu().numpy())
#             except Exception as e:
#                 print(f"⚠️ Failed to process {img_path}: {e}")
#                 continue
#         if embeds:
#             db_embeddings[identity] = np.mean(embeds, axis=0)  # mean embedding per identity

#     print(f" Collected {len(db_embeddings)} identity embeddings for {species_prefix}")

#     # Query prediction
#     query_metadata = species_metadata[species_metadata['split'] == 'query']
#     predictions = []
#     print(f"🔍 Predicting for query images of {species_prefix}...")

#     for _, row in tqdm(query_metadata.iterrows(), total=len(query_metadata)):
#         img_path = os.path.join(species_root, row['path'])
#         image_id = row['image_id']
#         try:
#             image = transform(Image.open(img_path).convert("RGB"))
#             image = image.unsqueeze(0).to(args.device)
#             with torch.no_grad():
#                 query_embed = model.extract_global_features(image)
#             if query_embed is None or query_embed[0] is None:
#                 predictions.append((image_id, "new_individual"))
#                 continue
#             query_embed_np = query_embed[0].squeeze().cpu().numpy()

#             # Compare with all DB identities
#             best_id = "new_individual"
#             best_sim = 0
#             for db_id, db_embed in db_embeddings.items():
#                 sim = cosine_similarity([query_embed_np], [db_embed])[0][0]
#                 if sim > best_sim:
#                     best_sim = sim
#                     best_id = db_id

#             if best_sim > threshold:
#                 predictions.append((image_id, best_id))
#             else:
#                 predictions.append((image_id, "new_individual"))

#         except Exception as e:
#             print(f"⚠️ Error processing query {img_path}: {e}")
#             predictions.append((image_id, "new_individual"))

#     return pd.DataFrame(predictions, columns=["image_id", "individual_id"])

import pickle

def match_and_generate_predictions_with_metadata(model, metadata_path, species_root, species_prefix, threshold=0.75):
    metadata = pd.read_csv(metadata_path)
    species_metadata = metadata[metadata['dataset'] == species_prefix]

    # ---- DB Embedding Cache Check ----
    cache_path = os.path.join(species_root, f"{species_prefix}_db_embeddings.pkl")
    if os.path.exists(cache_path):
        print(f"📦 Loading cached DB embeddings from: {cache_path}")
        with open(cache_path, 'rb') as f:
            db_embeddings = pickle.load(f)
    else:
        print(f"📁 Building database embeddings for {species_prefix}...")
        db_metadata = species_metadata[species_metadata['split'] == 'database']
        db_embeddings = {}

        for identity in tqdm(db_metadata['identity'].unique(), desc="DB identities"):
            identity_images = db_metadata[db_metadata['identity'] == identity]
            embeds = []
            for _, row in identity_images.iterrows():
                img_path = os.path.join(species_root, row['path'])
                if not os.path.exists(img_path):
                    continue
                try:
                    image = transform(Image.open(img_path).convert("RGB"))
                    image = image.unsqueeze(0).to(args.device)
                    with torch.no_grad():
                        embed = model.extract_global_features(image)
                    if embed is not None and embed[0] is not None:
                        embeds.append(embed[0].squeeze().cpu().numpy())
                except Exception as e:
                    print(f"⚠️ Failed to process {img_path}: {e}")
                    continue
            if embeds:
                db_embeddings[identity] = np.mean(embeds, axis=0)

        # Save cache
        with open(cache_path, 'wb') as f:
            pickle.dump(db_embeddings, f)
        print(f"✅ Cached DB embeddings to: {cache_path}")

    # ---- Query Prediction ----
    query_metadata = species_metadata[species_metadata['split'] == 'query']
    predictions = []
    print(f"🔍 Predicting for query images of {species_prefix}...")

    for _, row in tqdm(query_metadata.iterrows(), total=len(query_metadata)):
        img_path = os.path.join(species_root, row['path'])
        image_id = row['image_id']
        try:
            image = transform(Image.open(img_path).convert("RGB"))
            image = image.unsqueeze(0).to(args.device)
            with torch.no_grad():
                query_embed = model.extract_global_features(image)
            if query_embed is None or query_embed[0] is None:
                predictions.append((image_id, "new_individual"))
                continue
            query_embed_np = query_embed[0].squeeze().cpu().numpy()

            # Compare with all DB identities
            best_id = "new_individual"
            best_sim = 0
            # for db_id, db_embed in db_embeddings.items():
            #     sim = cosine_similarity([query_embed_np], [db_embed])[0][0]
            #     if sim > best_sim:
            #         best_sim = sim
            #         best_id = db_id

            # if best_sim > threshold:
            #     predictions.append((image_id, best_id))
            # else:
            #     predictions.append((image_id, "new_individual"))

            from scipy.special import softmax

            sims = cosine_similarity([query_embed_np], list(db_embeddings.values()))[0]
            db_ids = list(db_embeddings.keys())

            probabilities = softmax(sims / 0.05)
            max_idx = np.argmax(probabilities)

            if probabilities[max_idx] > 0.7:
                predictions.append((image_id, db_ids[max_idx]))
            else:
                predictions.append((image_id, "new_individual"))

        except Exception as e:
            print(f"⚠️ Error processing query {img_path}: {e}")
            predictions.append((image_id, "new_individual"))

    return pd.DataFrame(predictions, columns=["image_id", "individual_id"])



def main():
    print(f"🚀 Starting inference on GPU: {args.device}")
    model_path = 'tmp/rwod/best_model_Aerial_google_owlvit-base-patch16.pth'
    learned_path = 'tmp/rwod/fomo_learned_Aerial.pth'
    model, _ = load_model_for_inference(model_path, learned_path, args)

    root = '/home/aaditya23006/AMAN/SML/animal-clef-2025'
    metadata_path = os.path.join(root, 'metadata.csv')
    species_dirs = {
        'LynxID2025': 'submission_lynx.csv',
        'SalamanderID2025': 'submission_salamander.csv',
        'SeaTurtleID2022': 'submission_seaturtle.csv'
    }

    all_dfs = []
    for species, out_csv in species_dirs.items():
        print(f"\n🐾 Processing species: {species}")
        species_root = os.path.join(root)  # all paths in metadata.csv are relative to this
        df = match_and_generate_predictions_with_metadata(
            model=model,
            metadata_path=metadata_path,
            species_root=species_root,
            species_prefix=species,
            threshold=0.98
        )
        df.to_csv(out_csv, index=False)
        print(f"✅ Saved {species} predictions to: {out_csv}")
        all_dfs.append(df)

    # Combine all into final submission
    final_df = pd.concat(all_dfs, ignore_index=True)
    final_df.to_csv("final_submission_2.csv", index=False)
    print("\n🎯 Final combined submission saved as: final_submission.csv")

if __name__ == '__main__':
    main()

