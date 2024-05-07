from utils import PIPELINE
from pipelines import BaseVal

@PIPELINE.register()
def BaseTrain(train_dataset, 
              test_dataset, 
              optimizer, 
              scheduler, 
              model, 
              loss_function,
              logger,
              warmup=None,
              config=None,
              resume=None):
    starting_epoch = 1
    if resume is not None:
        starting_epoch = resume['EPOCH']
        model.load_state_dict(resume['MODEL'])
        optimizer.load_state_dict(resume['OPTIMIZER'])
        scheduler.load_state_dict(resume['SCHEDULER'])
        logger.EPOCH = starting_epoch
        logger.BestScore = resume['BESTSCORE']
    EPOCHS = config['settings']['EPOCHS']
    logger.data_time_start()
    for epoch in range(starting_epoch, EPOCHS+1):
        model.train()
        for batch_idx, (data, labels) in enumerate(train_dataset):
            optimizer.zero_grad()
            data = data.to('cuda')
            labels = labels.to('cuda')
            logger.data_time_end()
            logger.calc_time_start()
            outputs = model(data)
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
            if warmup is not None:
                if warmup.warmup_epochs >= epoch:
                    warmup.step()
        scheduler.step()
        if test_dataset is not None:
            BaseVal(test_dataset, model, logger)
        logger.data_time_start()