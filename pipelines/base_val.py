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

@PIPELINE.register()
def KDBaseVal(dataset,
              student,
              teacher,
              loss_function,
              logger):
    s_correct = 0
    t_correct = 0
    val_loss = 0
    logger.calc_time_start()
    student.eval()
    teacher.eval()
    for batch_idx, (image, labels) in enumerate(dataset):
        image = image.to('cuda')
        labels = labels.to('cuda')
        s_outputs = student(image)
        val_loss += loss_function(s_outputs[-1], labels).cpu().detach().item()
        s_outputs = s_outputs[-1]
        _, s_preds = s_outputs.max(1)
        s_correct += s_preds.eq(labels).sum()
        t_outputs = teacher(image)
        t_outputs = t_outputs[-1]
        _, t_preds = t_outputs.max(1)
        t_correct += t_preds.eq(labels).sum()
    logger.calc_time_end()
    logger.log_val(val_loss/len(dataset.dataset)*100, "loss")
    logger.log_val(s_correct.float()/len(dataset.dataset)*100, "accuracy/top1")
    logger.log_val(t_correct.float()/len(dataset.dataset)*100, "accuracy/top1_t")
    logger.update_val()