import torch
import numpy as np
from tqdm import tqdm

def generate_input_text(manufacturer, size):
    return f"{manufacturer} in the size interval {size} meters"

def generate_distractors_helper(true, possible, num_distractors=5):
    distractors = []
    # remove true from possible
    possible = [x for x in possible if x != true]
    for i in range(num_distractors):
        distractors.append(possible[np.random.randint(len(possible))])
    return distractors

def generate_distractors(true_manufacturer, true_size, possible_manufacturers, possible_sizes, num_distractors=10):
    distractors = []
    distractor_manufacturers = generate_distractors_helper(true_manufacturer, possible_manufacturers, num_distractors)
    distractor_sizes = generate_distractors_helper(true_size, possible_sizes, num_distractors)
    # pick the true manufacturer and true size
    distractors.append(generate_input_text(true_manufacturer, true_size))

    for i in range(num_distractors):
        # alternate between manufacturer and size
        if i % 2 == 0:
            distractors.append(generate_input_text(distractor_manufacturers[i], true_size))
        else:
            distractors.append(generate_input_text(true_manufacturer, distractor_sizes[i]))
    return distractors

def run_experiment_1(dataset, model, processor, device, batch_size=1,k=20, num_distractors=10):
    print("Running experiment 1...")
    # generate possible labels
    # possible_labels = generate_possible_labels(dataset)
    # prepare data collator
    top_manufacturers = np.unique(dataset["manufacturer"], return_counts=True)
    top_manufacturers = sorted(list(zip(top_manufacturers[0], top_manufacturers[1])), key=lambda x: x[1], reverse=True)
    top_manufacturers = [x[0] for x in top_manufacturers][:k]
    # find possible sizes for top manufacturers
    possible_sizes = dataset.filter(lambda x: x["manufacturer"] in top_manufacturers).unique("size category")
    def data_collator(batch):
        manufacturers = [x["manufacturer"] for x in batch]
        sizes = [x["size category"] for x in batch]
        images = [x["image"] for x in batch]
        labels = []
        # add distractors
        if batch_size == 1:
            labels += generate_distractors(manufacturers[0], sizes[0], top_manufacturers, possible_sizes, num_distractors=num_distractors)
        else:
            for i in range(len(batch)):
                distractors = generate_distractors(manufacturers[i], sizes[i], top_manufacturers, possible_sizes, num_distractors=num_distractors//len(batch))
                labels += distractors
                        # labels = [f"{manufacturer} in the size interval {size} meter" for manufacturer, size in zip(manufacturers, sizes)]
        # shuffle labels
        np.random.shuffle(labels)
        # prepare inputs
        inputs = processor(text=labels, images=images, return_tensors="pt", padding=True)
        inputs.to(device)
        return inputs
        


    # prepare dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=data_collator)
    # prepare model
    model = model.to(device)

    # run the experiment
    correct = 0
    total = 0
    for batch in tqdm(dataloader):
        outputs = model(**batch)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        probs = probs.detach().to("cpu").numpy()
        correct += np.sum(np.argmax(probs, axis=1) == 0)
        total += len(probs)
    return correct / total

# # run the experiment
# accuracy = run_experiment_1(dataset, CLIP_not_FT, processor, device)



# def generate_possible_labels(dataset,k = 20):
#     manufacturer_counts = np.unique(dataset["manufacturer"], return_counts=True)
#     manufacturer_counts = list(zip(manufacturer_counts[0], manufacturer_counts[1]))
#     manufacturer_counts = sorted(manufacturer_counts, key=lambda x: x[1], reverse=True)
#     # take top 20 manufacturers
#     top_manufacturers = manufacturer_counts[:k]
#     # only keep manufacturer name
#     top_manufacturers = [x[0] for x in top_manufacturers]
#     # calculate possible sizes for top manufacturers
#     top_manufacturer_dataset = dataset.filter(lambda x: x["manufacturer"] in top_manufacturers)
#     possible_sizes = np.unique(top_manufacturer_dataset["size category"])
#     # make a list of combinations
#     possible_labels = []
#     for manufacturer in top_manufacturers:
#         for size in possible_sizes:
#             possible_labels.append(generate_input_text(manufacturer, size))
#     return possible_labels
