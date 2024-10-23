def get_model(model_name, args):
    name = model_name.lower()
    
    if name == "ease":
        from models.ease import Learner
    elif name == "dsease":
        from models.dsease import Learner
    else:
        assert 0
    return Learner(args)