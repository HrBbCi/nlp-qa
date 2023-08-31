from transformers import TrainingArguments, Trainer
from model.qa_model import MRCQA
from utils import data_loader
from configs.config import *
#os.environ["CUDA_VISIBLE_DEVICES"] = ""
if __name__ == "__main__":
    # tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    model = MRCQA.from_pretrained(MODEL_PRETRAIN,
                                                 cache_dir='{}/cache'.format(PATH_CACHE_LOG),
                                                 #local_files_only=True
                                                )
    print(model)
    print(model.config)

    train_dataset, valid_dataset = data_loader.get_dataloader(
        train_path='./data-bin/processed/train.dataset',
        valid_path='./data-bin/processed/valid.dataset'
    )
    ep = 1
    training_args = TrainingArguments(PATH_TRAINING_CHECKPOINT,
                                    do_train=True,
                                    do_eval=True,
                                    num_train_epochs=10,
                                    learning_rate=1e-5,
                                    warmup_ratio=0.05,
                                    weight_decay=0.01,
                                    per_device_train_batch_size=1,
                                    per_device_eval_batch_size=1,
                                    gradient_accumulation_steps=1,
                                    logging_dir= PATH_TRAINING_LOG,
                                    logging_steps=1,
                                    label_names=['start_positions',
                                                'end_positions',
                                                'span_answer_ids',
                                                'input_ids',
                                                'words_lengths'],
                                    group_by_length=True,
                                    save_strategy="epoch",
                                    metric_for_best_model='f1',
                                    load_best_model_at_end=True,
                                    save_total_limit=2,
                                    evaluation_strategy="epoch",
                                    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_loader.data_collator,
        compute_metrics=data_loader.compute_metrics
    )

    trainer.train()

