

### Dataload iterator
## class dataload:
##   def ...
### init => créer un tableau (class_num, class_name) where class_num in {0..100}
###      => créer tableau "images" qui liste toutes les images (chemin vers l'image de food 101, classe associée)
### run => récupérer random batch_size images dans son tableau "images", et return (inputs,targets)

## batch_size = 20
## DataLoader = dataload(batch_size)
## for batch in DataLoader:
##    inputs, targets = batch
### inputs est un array de dimension (bach_size, 3, width, height)
### targets est un array de dimension (batch_size,)
##    print inputs.size()
