from utlis import MODEL

def build_model(opt):
    TYPE = opt['TYPE']
    opt.pop('TYPE')
    if isinstance(opt, dict):
        model = MODEL.get(TYPE)(**opt)
    elif isinstance(opt, str):
        model = MODEL.get(TYPE)()
    elif isinstance(opt, list):
        model = MODEL.get(TYPE)(*opt)
    return model

def build_pipeline(opt):
    TYPE = opt['TYPE']
    opt.pop('TYPE')
    if isinstance(opt, dict):
        pipeline = PIPELINE.get(TYPE)(**opt)
    elif isinstance(opt, str):
        pipeline = PIPELINE.get(TYPE)()
    elif isinstance(opt, list):
        pipeline = PIPLINE.get(TYPE)(*opt)
    return pipeline