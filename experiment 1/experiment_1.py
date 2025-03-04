import torch
import numpy as np
from tqdm import tqdm

def parse_size_interval(size):
    # example "[14.5-15.5)" -> "14.5 to 15.5 metes"
    from_, to_ = size[1:-1].split("-")
    # trim any ")" or "[" from the string
    from_ = from_.strip("([")
    to_ = to_.strip(")]")
    return f"{from_} to {to_} meters"

def generate_input_text(manufacturer, size):
    return f"{manufacturer} in the size interval {parse_size_interval(size)} meters"

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

def calculate_accuracy(probs, candidate_labels : list[str], correct_label : str):
    # find the correct label
    correct_label_index = candidate_labels.index(correct_label)
    return np.argmax(probs, axis=1) == correct_label_index

def run_experiment_1(dataset, model, processor, device, batch_size=1,k=20, num_distractors=10):
    print("Running experiment 1...")
    # generate possible labels
    # possible_labels = generate_possible_labels(dataset)
    # prepare data collator
    top_manufacturers = np.unique(dataset["manufacturer"], return_counts=True)
    top_manufacturers = sorted(list(zip(top_manufacturers[0], top_manufacturers[1])), key=lambda x: x[1], reverse=True)
    top_manufacturers = [x[0] for x in top_manufacturers][:k]
    dataset = dataset.filter(lambda x: x["manufacturer"] in top_manufacturers)
    # find possible sizes for top manufacturers
    possible_sizes = dataset.unique("size category")
    def data_collator(batch):
        manufacturers = [x["manufacturer"] for x in batch]
        sizes = [x["size category"] for x in batch]
        images = [x["image"] for x in batch]
        labels = []
        correct_labels = []
        # add distractors
        if batch_size == 1:
            labels += generate_distractors(manufacturers[0], sizes[0], top_manufacturers, possible_sizes, num_distractors=num_distractors)
            correct_labels = (generate_input_text(manufacturers[0], sizes[0]))
        else:
            for i in range(len(batch)):
                distractors = generate_distractors(manufacturers[i], sizes[i], top_manufacturers, possible_sizes, num_distractors=num_distractors//len(batch))
                labels += distractors
                correct_labels.append(generate_input_text(manufacturers[i], sizes[i]))
                        # labels = [f"{manufacturer} in the size interval {size} meter" for manufacturer, size in zip(manufacturers, sizes)]
        # shuffle labels
        np.random.shuffle(labels)
        # prepare inputs
        inputs = processor(text=labels, images=images, return_tensors="pt", padding=True)
        inputs.to(device)
        return (inputs, labels, correct_labels)
    
    # prepare dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=data_collator)
    # prepare model
    model = model.to(device)

    # run the experiment
    correct = 0
    total = 0
    for batch in tqdm(dataloader):
        inputs, candidate_labels, correct_label = batch
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        probs = probs.detach().to("cpu").numpy()

        correct += np.sum(calculate_accuracy(probs, candidate_labels, correct_label))
        total += len(probs)
    return correct / total

def run_experiment_1_b(dataset, model, processor, device, batch_size=1, k=20):
    # same as experiment 1 but only with manufacturer and not size
    top_manufacturers = np.unique(dataset["manufacturer"], return_counts=True)
    top_manufacturers = sorted(list(zip(top_manufacturers[0], top_manufacturers[1])), key=lambda x: x[1], reverse=True)
    top_manufacturers = [x[0] for x in top_manufacturers][:k]
    dataset = dataset.filter(lambda x: x["manufacturer"] in top_manufacturers)
    def data_collator(batch):
        images = [x["image"] for x in batch]
        inputs = processor(text = top_manufacturers, images=images, return_tensors="pt", padding=True)
        inputs.to(device)
        if batch_size == 1:
            return (inputs, top_manufacturers, batch[0]["manufacturer"])
        else:
            raise NotImplementedError("Batch size > 1 not supported for this experiment")
    # prepare dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=data_collator)
    # prepare model
    model = model.to(device)
    # run the experiment
    correct = 0
    total = 0
    for batch in tqdm(dataloader):
        inputs, candidate_labels, correct_label = batch
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        probs = probs.detach().to("cpu").numpy()

        correct += np.sum(calculate_accuracy(probs, candidate_labels, correct_label))
        total += len(probs)
    return correct / total

def run_experiment_1_c(dataset, model, processor, device, batch_size=1, k=20):
    top_manufacturers = np.unique(dataset["manufacturer"], return_counts=True)
    top_manufacturers = sorted(list(zip(top_manufacturers[0], top_manufacturers[1])), key=lambda x: x[1], reverse=True)
    top_manufacturers = [x[0] for x in top_manufacturers][:k]
    dataset = dataset.filter(lambda x: x["manufacturer"] in top_manufacturers)
    # same as experiment 1 but only with size and not manufacturer
    possible_sizes = dataset.unique("size category")
    # apply parse_size_interval to all sizes
    possible_sizes = [parse_size_interval(size) for size in possible_sizes]
    def data_collator(batch):
        images = [x["image"] for x in batch]
        inputs = processor(text = possible_sizes, images=images, return_tensors="pt", padding=True)
        inputs.to(device)
        if batch_size == 1:
            return (inputs, possible_sizes, parse_size_interval(batch[0]["size category"]))
        else:
            raise NotImplementedError("Batch size > 1 not supported for this experiment")
    # prepare dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=data_collator)
    # prepare model
    model = model.to(device)
    # run the experiment
    correct = 0
    total = 0
    for batch in tqdm(dataloader):
        inputs, candidate_labels, correct_label = batch
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        probs = probs.detach().to("cpu").numpy()

        correct += np.sum(calculate_accuracy(probs, candidate_labels, correct_label))
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
