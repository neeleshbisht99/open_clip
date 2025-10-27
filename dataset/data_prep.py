import os
import random
import csv

random.seed(42)

def load_images(images_metadata_file, images_folder):
    image_id_to_path = {}
    with open(images_metadata_file, 'r') as f:
        for line in f:
            image_id, image_path = line.strip().split()
            image_id = int(image_id)
            image_id_to_path[image_id] = os.path.join(images_folder, image_path)
    return image_id_to_path


def load_image_train_test(train_test_metadata_file):
    image_id_to_train_test = {}
    with open(train_test_metadata_file, 'r') as f:
        for line in f:
            image_id, is_training_img = line.strip().split()
            image_id = int(image_id)
            image_id_to_train_test[image_id] = is_training_img == '1'
    return image_id_to_train_test


def load_classes(classes_metadata_file):
    class_id_to_name = {}
    class_id_to_name_clean = {}
    with open(classes_metadata_file, 'r') as f:
        for line in f:
            class_id, class_name = line.strip().split()
            class_id = int(class_id)
            class_id_to_name[class_id] = class_name.lower()
            class_id_to_name_clean[class_id] = class_name.split('.')[1].replace('_', ' ')
    return class_id_to_name, class_id_to_name_clean

def load_image_class(image_class_labels_metadata_file):
    image_id_to_class_id = {}
    with open(image_class_labels_metadata_file, 'r') as f:
        for line in f:
            image_id, class_id = line.strip().split()
            image_id = int(image_id)
            class_id = int(class_id)
            image_id_to_class_id[image_id] = class_id
    return image_id_to_class_id


def load_attributes(attributes_metadata_file):
    attribute_id_to_name = {}
    with open(attributes_metadata_file, 'r') as f:
        for line in f:
            attribute_id, attribute_name = line.strip().split()
            attribute_id = int(attribute_id)
            attribute_id_to_name[attribute_id] = attribute_name
    return attribute_id_to_name

def load_image_attributes(image_attributes_metadata_file):
    image_id_to_attributes = {}
    with open(image_attributes_metadata_file, 'r') as f:
        for line in f:
            image_id, attribute_id, is_present, certainty_id, _ = line.strip().split()
            image_id = int(image_id)
            certainty_id = int(certainty_id)
            attribute_id = int(attribute_id)
            is_present = int(is_present)
            if is_present and certainty_id > 2:
                if image_id not in image_id_to_attributes: image_id_to_attributes[image_id] = set()
                image_id_to_attributes[image_id].add(attribute_id)
    return image_id_to_attributes

# -----------------------
# Prompt templates
# -----------------------
WITH_ATTR_TEMPLATES = [
    "a photo of {cls} with {attrs}.",
    "a close-up of {cls} showing {attrs}.",
    "this is a {cls} that has {attrs}.",
    "a wild {cls} featuring {attrs}.",
    "an image of the bird species {cls} with {attrs}.",
]

CLASS_ONLY_TEMPLATES = [
    "a photo of a {cls}.",
    "this is an {cls}.",
    "an image of the bird species {cls}.",
]

def caption_generator(image_id, image_id_to_class_id, class_id_to_name_clean, image_id_to_attributes, attribute_id_to_name) -> list[str]:
    class_id = image_id_to_class_id[image_id]
    class_name = class_id_to_name_clean[class_id]
    image_attribute_ids = image_id_to_attributes.get(image_id, set())
    image_attributes = []
    for attr in list(image_attribute_ids):
        image_attributes.append(attribute_id_to_name[attr])
    
    n_attrs = len(image_attributes)
    
    attr_template_idxs = random.sample(range(0, 5), 2)
    class_template_idx = random.randint(0, 2)

    captions = []

    cls_temp = CLASS_ONLY_TEMPLATES[class_template_idx]
    cls_temp = cls_temp.format(cls=class_name)
    captions.append(cls_temp)

    for idx in attr_template_idxs:
        num_attrs = random.randint(min(n_attrs,3), min(n_attrs, 7))
        if(num_attrs == 0):
            captions.append(cls_temp)
            continue
        selected_image_attributes = random.sample(image_attributes, num_attrs)
        final_image_attributes = []
        for img_attr in selected_image_attributes:
            key, val = img_attr.split('::')
            phrase = ' '.join(key.split('_')[1:])
            phrase = phrase.replace('(', '').replace(')', '').replace('_', ' ')
            final_image_attributes.append(f"{val.replace('_',' ')} {phrase}")

        attrs = ' and '.join(final_image_attributes)
        temp = WITH_ATTR_TEMPLATES[idx].format(cls=class_name, attrs=attrs)
        captions.append(temp)
    
    return captions

def create_csv(file_name, selected_class_ids, image_id_to_path, image_id_to_class_id, class_id_to_name_clean, image_id_to_attributes, attribute_id_to_name, image_id_to_train_test=None, require_train=None):
    header = ['filepath','caption']
    rows = []
    for image_id, path in image_id_to_path.items():
        class_id = image_id_to_class_id[image_id]
        if class_id not in selected_class_ids: 
            continue
        if image_id_to_train_test is not None:
            is_train_img = image_id_to_train_test.get(image_id, True)
            if require_train != is_train_img:
                continue
        captions = caption_generator(image_id, image_id_to_class_id, class_id_to_name_clean, image_id_to_attributes, attribute_id_to_name)
        for cap in captions:
            rows.append([path, cap])
    
    with open(file_name, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)
    
    print(f"Data saved to {file_name} successfully!")


def main():
    root_path = "/shared/scratch/0/home/v_neelesh_bisht/projects/open_clip/dataset/CUB_200_2011"
    
    image_id_to_path = load_images(os.path.join(root_path, 'images.txt'), os.path.join(root_path, 'images'))
    print("Created image_id_to_path \n")
    
    image_id_to_train_test = load_image_train_test(os.path.join(root_path, 'train_test_split.txt'))
    print("Created image_id_to_train_test \n")
    
    class_id_to_name, class_id_to_name_clean = load_classes(os.path.join(root_path, 'classes.txt'))
    print("Created class_id_to_name_clean \n")
    
    image_id_to_class_id = load_image_class(os.path.join(root_path, 'image_class_labels.txt'))
    print("Created image_id_to_class_id \n")
    
    attribute_id_to_name = load_attributes(os.path.join(root_path, 'attributes/attributes.txt'))
    print("Created attribute_id_to_name \n")
    
    image_id_to_attributes = load_image_attributes(os.path.join(root_path, 'attributes/image_attribute_labels_clean.txt'))
    print("Created image_id_to_attributes \n")
    
    num_classes = 100
    train_selected_class_ids = set(random.sample(range(1, 201), num_classes))
    eval_selected_class_ids = [i for i in range(1,201) if i not in set(train_selected_class_ids)]

    with open('train_selected_class_ids.txt','w') as f:
        for id in sorted(train_selected_class_ids):
            f.write(f"{id}\n")
    
    with open('eval_selected_class_ids.txt','w') as f:
        for id in sorted(eval_selected_class_ids):
            f.write(f"{id}\n")

    create_csv('cub_finetune_100cls_train.csv', train_selected_class_ids, image_id_to_path, image_id_to_class_id, class_id_to_name_clean, image_id_to_attributes, attribute_id_to_name, image_id_to_train_test=image_id_to_train_test, require_train=True)
    create_csv('cub_finetune_100cls_val.csv', train_selected_class_ids, image_id_to_path, image_id_to_class_id, class_id_to_name_clean, image_id_to_attributes, attribute_id_to_name, image_id_to_train_test=image_id_to_train_test, require_train=False)
    create_csv('cub_finetune_100cls_test.csv', eval_selected_class_ids, image_id_to_path, image_id_to_class_id, class_id_to_name_clean, image_id_to_attributes, attribute_id_to_name, image_id_to_train_test = None, require_train=None)

if __name__ == "__main__":
    main()