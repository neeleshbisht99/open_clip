1. local dataset must include both images and their corresponding textual descriptions.

2. Create an index file that links the images with their respective captions.

3. For local data, using a CSV file as an index is the most convenient option.

4. Format:
   filepath,caption
   /base_path/img/Party Penguins_6664.png,"A picture of Party Penguins, containing Red Background Stitches Cheeks Cute Eyes Normal Beak None Face Basketball Hat Red Jacket Clothes."

Info out dataset:
1. images.txt: image_id, image_path
2. train_test_split.txt: image_id, is_training_image

3. classes.txt: class_id, class_name
4. image_class_labels.txt: image_id, class_id

5. attributes.txt: attribute_id, attribute_name
6. certainties.txt: certainty_id, certainty_name
7. image_attribute_labels_clean.txt: image_id, attribute_id, is_present, certainty_id, time
8. class_attribute_labels_continuous.txt: class_id, {...312 attribute column with percentage of presence yes/no by workers}




