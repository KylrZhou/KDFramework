from utils import PIPELINE
from pipelines import KDBaseVal

@PIPELINE.register()
def KDBaseTrain(train_dataset,
                test_dataset, 
                optimizer, 
                scheduler, 
                model, 
                loss_function,
                logger, 
                distiller,
                config):
    EPOCHS = config['settings']['EPOCHS']
    ALPHA = config['distiller']['ALPHA']
    logger.data_time_start()
    for epoch in range(1,EPOCHS+1):
        model.train()
        for batch_idx, (data, labels) in enumerate(train_dataset):
            optimizer.zero_grad()
            data = data.to('cuda')
            labels = labels.to('cuda')
            logger.data_time_end()
            logger.calc_time_start()
            outputs = model(data)
            label_loss = logger.log(loss_function(outputs[-1], labels), "LabelLoss")
            kd_loss = distiller.distill(data, labels, model)
            loss = label_loss * ALPHA + kd_loss
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
            KDBaseVal(test_dataset, model, distiller.teacher, logger)
        logger.data_time_start()