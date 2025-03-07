# Knowledge Distillation from VGG16 to MobileNetV2

- Teacher model - VGG16
- Student model - MobileNetV2

### 3. Train the model
To train the VGG16 model
```bash
# use gpu to train vgg16
$ python train.py -net vgg16 -gpu
```

To train the Mobilenet model
```bash
# use gpu to train mobilenet
$ python train.py -net mobilenet -gpu
```

To perform knowledge distillation from the trained VGG16 to the Mobilenet model
```bash
# use gpu to train mobilenet
$ python knowledge_distillation_train.py -gpu -teacher path_to_best_vgg16_weights_file -student path_to_best_mobilenet_weights_file
```

The weights file with the best accuracy would be written to the disk with name suffix 'best' (default in checkpoint folder).


### 4. test the model
Test the VGG16 model 
```bash
$ python test.py -net vgg16 -weights path_to_best_vgg16_weights_file
```

Test the mobilenet model 
```bash
$ python test.py -net mobilenet -weights path_to_best_mobilenet_weights_file
```

Test the knowledge distilled mobilenet model 
```bash
$ python knowledge_distillation_test.py -gpu -weights path_to_best_knowledge_distilled_mobilenet_weights_file
```
