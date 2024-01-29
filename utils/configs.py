def getConfig(specification):
    match (specification):
        case _:  
            return {
               "epochs": 10,
                "labelled_batch_size": 32,
                "unlabelled_batch_size": 32,
                "num_classes": 2,
                "lr": 1e-5,
                "T1": 1,
                "T2": 6,
                "alpha_f": .03,
            }

