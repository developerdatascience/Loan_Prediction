import mlflow


class IntiateMLflow:
    def __init__(self,
                 experiment_name,
                 run_name,
                 run_metrics,
                 confusion_matrix_path,
                 roc_auc_plot_path = None,
                 run_params = None) -> None:
        self.experiment_name = experiment_name,
        self.run_name = run_name,
        self.run_metrics = run_metrics,
        self.confusion_matrix_path = confusion_matrix_path,
        self.roc_auc_plot_path = roc_auc_plot_path,
        self.run_params = run_params
        
        def create_experiment(self):
            pass