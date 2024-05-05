from utils import PIPELINE

@PIPELINE.register()
def BaseVal(dataset, 
            model, 
            logger):
    correct = 0
    logger.calc_time_start()
    model.eval()
    for batch_idx, (image, labels) in enumerate(dataset):
        image = image.to('cuda')
        labels = labels.to('cuda')
        outputs = model(image)
        outputs = outputs[-1]
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()
    logger.calc_time_end()
    logger.log_val(correct.float()/len(dataset.dataset)*100, "Acc")
    logger.update_val()