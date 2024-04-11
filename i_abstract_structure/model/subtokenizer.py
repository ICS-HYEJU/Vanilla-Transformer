import sentencepiece as spm

def load_subtokenizer(corpus_src, corpus_trg, vocab_size):
    import os
    current_path = os.getcwd()

    # make trained model
    spm.SentencePieceTrainer.train(
        f"--input={corpus_src} --model_prefix=src --vocab_size={vocab_size}" +
        " --model_type=bpe" +
        " --max_sentence_length=999999" +
        " --pad_id=0 --pad_piece=[PAD]" +  # pad (0)
        " --unk_id=1 --unk_piece=[UNK]" +  # unknown (1)
        " --bos_id=2 --bos_piece=[BOS]" +  # begin of sequence (2)
        " --eos_id=3 --eos_piece=[EOS]" +  # end of sequence (3)
        " --user_defined_symbols=[SEP],[CLS],[MASK]")

    spm.SentencePieceTrainer.train(
        f"--input={corpus_trg} --model_prefix=trg --vocab_size={vocab_size}" +
        " --model_type=bpe" +
        " --max_sentence_length=999999" +
        " --pad_id=0 --pad_piece=[PAD]" +  # pad (0)
        " --unk_id=1 --unk_piece=[UNK]" +  # unknown (1)
        " --bos_id=2 --bos_piece=[BOS]" +  # begin of sequence (2)
        " --eos_id=3 --eos_piece=[EOS]" +  # end of sequence (3)
        " --user_defined_symbols=[SEP],[CLS],[MASK]")

    sp_src = spm.SentencePieceProcessor()
    sp_src.Load(os.path.join(current_path, 'src.model'))

    sp_trg = spm.SentencePieceProcessor()
    sp_trg.Load(os.path.join(current_path, 'trg.model'))
    #
    return sp_src, sp_trg

