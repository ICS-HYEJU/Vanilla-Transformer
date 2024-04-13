def get_config_dict():
    dataset_info = dict(
        name = 'bible',
        path='/storage/hjchoi/T_dataset/',
        #
        vocab_size = 10000,
        seq_len = 60,
        batch_size = 64,
        #
        hidden_dim=256,
        num_head=8,
        inner_dim=512,
        N=2,
        #
        PAD_IDX = 0,
        BOS_IDX = 2,
        EOS_IDX = 3,
        #
    )

    path = dict(
        save_base_path = 'runs'
    )

    subtokenizer = dict(
        # load model
        en='/home/hjchoi/PycharmProjects/transformer/i_abstract_structure/model/trg.model',
        ko='/home/hjchoi/PycharmProjects/transformer/i_abstract_structure/model/src.model',
        # load dataset(.txt)
        en_corpus='/home/hjchoi/PycharmProjects/transformer/i_abstract_structure/dataset/src.txt',
        ko_corpus='/home/hjchoi/PycharmProjects/transformer/i_abstract_structure/dataset/trg.txt'
    )

    model = dict(
        name = 'Transformer'
    )

    solver = dict(
        name = 'Adam',
        gpu_id = 0,
        lr0 = 1e-4,
        weight_decay = 5e-4,
        max_epoch = 10
    )

    scheduler = dict(
        name ='CosineAnnealingLR',
        T_max = 100,
        eta_min = 1e-5,

    )
    weight_info = dict(
        name = 'last_weight.pth',
    )

    # Merge all information into a dictionary variable
    config = dict(
        dataset_info = dataset_info,
        path = path,
        subtokenizer = subtokenizer,
        model = model,
        solver = solver,
        scheduler= scheduler,
        weight_info = weight_info
    )

    return config