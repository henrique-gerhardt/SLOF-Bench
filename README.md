Credits:

        https://github.com/haruishi43/equilib
        
        https://github.com/princeton-vl/RAFT
        
        https://github.com/sammy-su/KernelTransformerNetwork
        
        https://github.com/facebookresearch/simsiam
        
        https://www.crowd-render.com/


# Learning Omnidirectional Flow via Simple Siamese Representation

### Use main.py
### to train on single rotation
    # model to load
        # final_cache/
            # doublerotation.pt  ktn.pt  raftfinetune.pt  raft.pt  singlerotation.pt  switchrotation.pt
            
            # note
                # train mode was enable only for
                    # singlerotation, switchrotation, doublerotation, raftfinetune, raft

                # use finetune True for following
                    # raftfinetune, raft, ktn
            
        
        # python main.py --clip_grad --load

        # Note data directory
            - FLOW360_train_test
                - test
                    - 001
                        -bflows
                            - 0001.npy
                            - 0002.npy
                            ..
                            - n-1.npy
                        -fflows
                            - 0001.npy
                            - 0002.npy
                            ..
                        -frames
                            - 0001.png
                            - 0002.png
                            ..
                            - n.png
                    - 002
                    ..
                - train
                    - 001
                    - 002
                    ..

### Please see all options in main.py

### Download test data from the link provided in supplimental
