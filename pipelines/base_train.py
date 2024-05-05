from utils import PIPELINE
from pipelines import BaseVal

@PIPELINE.register()
def BaseTrain(dataset, 
              test_dataset, 
              optimizer, 
              scheduler, 
              model, 
              loss_function,
              logger, 
              config=None, 
              distiller=None):
    logger.data_time_start()
    for epoch in range(1,201):
        model.train()
        for batch_idx, (image, labels) in enumerate(dataset):
            optimizer.zero_grad()
            image = image.to('cuda')
            labels = labels.to('cuda')
            logger.data_time_end()
            logger.calc_time_start()
            outputs = model(image)
            outputs = outputs[-1]
            loss = logger.log(loss_function(outputs, labels), "Loss")
            loss.backward()
            optimizer.step()
            logger.calc_time_end()
            logger.data_time_start()
            n_iter = batch_idx + 1
            lr = logger.log(optimizer.param_groups[0]['lr'], "lr")
            log_dict = {"Loss":loss,"lr":lr}
            logger.update()
        scheduler.step()
        if test_dataset is not None:
            BaseVal(test_dataset, model, logger)
        logger.data_time_start()
