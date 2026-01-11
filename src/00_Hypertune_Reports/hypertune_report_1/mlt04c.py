modeldir = Path("models").resolve()
if not modeldir.exists():
    modeldir.mkdir()
    print(f"Created {modeldir}")

import torch.optim as optim
from mltrainer import metrics, Trainer, TrainerSettings, ReportTypes
from datetime import datetime

# Define the hyperparameter search space
settings = TrainerSettings(
    epochs=3,
    metrics=[accuracy],
    logdir=modeldir,
    train_steps=100,
    valid_steps=100,
    reporttypes=[ReportTypes.MLFLOW, ReportTypes.TOML],
)

# Define the objective function for hyperparameter optimization
def objective(params):
    # Start a new MLflow run for tracking the experiment
    with mlflow.start_run():
        # Set MLflow tags to record metadata about the model and developer
        mlflow.set_tag("model", "convnet")
        mlflow.set_tag("dev", "raoul")
        # Log hyperparameters to MLflow
        mlflow.log_params(params)
        mlflow.log_param("batchsize", f"{batchsize}")


        # Initialize the optimizer, loss function, and accuracy metric
        optimizer = optim.Adam
        loss_fn = torch.nn.CrossEntropyLoss()
        accuracy = metrics.Accuracy()
        config = CNNConfig(
            matrixshape = (28, 28), # every image is 28x28
            batchsize = batchsize,
            input_channels = 1, # we have black and white images, so only one channel
            hidden = params["filters"], # number of filters
            kernel_size = 3, # kernel size of the convolution
            maxpool = 3, # kernel size of the maxpool
            num_layers = 4, # we will stack 4 Convolutional blocks, each with two Conv2d layers
            num_classes = 10,
        )

        # Instantiate the CNN model with the given hyperparameters
        model = CNNblocks(config)
        # Train the model using a custom train loop
        trainer = Trainer(
            model=model,
            settings=settings,
            loss_fn=loss_fn,
            optimizer=optimizer,
            traindataloader=trainstreamer,
            validdataloader=validstreamer,
            scheduler=optim.lr_scheduler.ReduceLROnPlateau,
            device=device,
        )
        trainer.loop()

        # Save the trained model with a timestamp
        tag = datetime.now().strftime("%Y%m%d-%H%M")
        modelpath = modeldir / (tag + "model.pt")
        torch.save(model, modelpath)

        # Log the saved model as an artifact in MLflow
        mlflow.log_artifact(local_path=modelpath, artifact_path="pytorch_models")
        return {'loss' : trainer.test_loss, 'status': STATUS_OK}

search_space = {
    'filters' : scope.int(hp.quniform('filters', 16, 128, 8)),
    'kernel_size' : scope.int(hp.quniform('kernel_size', 2, 5, 1)),
    'num_layers' : scope.int(hp.quniform('num_layers', 1, 10, 1)),
}

best_result = fmin(
    fn=objective,
    space=search_space,
    algo=tpe.suggest,
    max_evals=3,
    trials=Trials()
)

print("best_result: "+str(best_result)