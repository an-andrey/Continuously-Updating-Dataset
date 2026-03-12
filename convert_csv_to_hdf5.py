    import h5py

    with h5py.File("data/midjourney_prompts.h5", "w"):
        dataset = h5py.Dataset("prompts", data="data/openai_Udxg6_etsy_prompts.csv")
