from openai import OpenAI
import os
import shap
import base64
import matplotlib.pyplot as plt
import io
from pydantic import BaseModel


# Function to encode the image
def encode_image(image):
    image.seek(0)
    return base64.b64encode(image.read()).decode('utf-8')

# Function to encode the image
def encode_image_path(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

class Explainer:
    def __init__(
        self,
        dataset_context,
        model_task_context,
        model,
        X_dataset,
        features_descriptions=None,
        shap_algorithm="auto",
        scatter_max_samples=1000,
        llm_client=None,
        llm_model="gpt-4o-mini",
    ):
        """
        Args:
            dataset_context (string): Brief description of the dataset
                (ie. "## Daily Bike-sharing rental process data set is related to the two-year historical log corresponding to years 2011 and 2012 from Capital Bikeshare system, Washington D.C., USA")
            model_task_context (string): Brief description of what the model predicts
                (ie. "the model predicts total daily count of rental bikes including both casual and registered")
            model (object or function): User supplied function or model object that takes a dataset of samples and
            computes the output of the model for those samples.
            X_dataset (pandas.DataFrame): the dataset
            features_descriptions (dictionary (string>string): Brief description of each feature. Key is feature name.
            shap_algorithm (string): "auto", "permutation", "partition", "tree", or "linear"
            scatter_max_samples (int): max samples for shap scatter plot
            llm_client (LLM client object):  LLM object to use.
            should be able to get image as input
            llm_model (string)

        This function uses the following environment variables:

        - OPENAI_API_KEY
        - OPENAI_API_ORG
        """
        self.llm_model = llm_model
        self.llm_client = llm_client
        if self.llm_client is None:
            self.llm_client = OpenAI(
            api_key=(
                os.environ["OPENAI_API_KEY"] if "OPENAI_API_KEY" in os.environ else None
            ),
            organization=(
                os.environ["OPENAI_API_ORG"] if "OPENAI_API_ORG" in os.environ else None
            ))

        self.dataset_context = dataset_context
        self.model_task_context = model_task_context
        self.features_descriptions = features_descriptions
        self.model = model
        self.X_dataset = X_dataset
        self.shap_algorithm = shap_algorithm
        self.shap_values = None
        self.llm_cache = {}
        self.scatter_max_samples = scatter_max_samples

    def describe_graph_assemble_prompt(self,
        feature_name: str,
        num_sentences: int = 7,
        show_plot=False):

        if self.shap_values is None:
            if self.shap_algorithm == "tree":
                shap_explainer = shap.TreeExplainer(self.model)
            else:
                background = shap.maskers.Independent(self.X_dataset, max_samples=100)
                shap_explainer = shap.Explainer(self.model, background, algorithm=self.shap_algorithm)
            self.shap_values = shap_explainer(self.X_dataset.head(self.scatter_max_samples))

        image = io.BytesIO()
        feature_index = list(self.X_dataset.columns).index(feature_name)
        shap.plots.scatter(self.shap_values[:, feature_index], show=show_plot)
        #plt.savefig(f"scatter_shap_{feature_name}.png", format='png', bbox_inches="tight")
        plt.savefig(image, format='png', bbox_inches="tight")
        plt.close()

        base64_image_graph = encode_image(image)

        return  self.prompt_describe_graph_cot(
            base64_image_graph, feature_name, num_sentences=num_sentences
        )

    def describe_graph(
        self,
        feature_name: str,
        num_sentences: int = 7,
        show_plot=False
    ):
        """Ask the LLM to describe a graph. Uses chain-of-thought reasoning.

        Args:
            llm (Union[AbstractChatModel, str]): The LLM.
            ebm (Union[ExplainableBoostingClassifier, ExplainableBoostingRegressor]): The EBM.
            feature_name (str): The name of the feature to describe.
            num_sentences (int, optional): The desired number of senteces for the description. Defaults to 7.
            show_plot: if to show plot

        Returns:
            str:  The description of the graph.
        """
        messages = self.describe_graph_assemble_prompt(feature_name, num_sentences, show_plot=show_plot)
        """
        # execute the prompt
        completion = self.llm_client.chat.completions.create(
            model=self.llm_model,
            messages=messages
        )

        return completion.choices[0].message.content
        """
        class Task(BaseModel):
            explanation: str
            output: str

        class TaskReasoning(BaseModel):
            tasks: list[Task]
            final_answer: str

        completion = self.llm_client.beta.chat.completions.parse(
            model=self.llm_model,
            messages=messages,
            response_format=TaskReasoning,
        )

        return completion.choices[0].message.parsed

    def prompt_describe_graph(
            self,
            base64_image_graph: str,
            feature_name
    ):
        """Prompt the LLM to describe a graph. This is intended to be the first prompt in a conversation about a graph.

        Args:
            base64_image_graph (str): The graph to describe. byte64 encoded image.
            feature_name (str): Feature name
        Returns:
            Messages in OpenAI format.
        """
        prompt_text = """
            You interpret dependence scatter plots.
            A dependence scatter plot shows the effect a single feature has on the predictions made by the model.
            "Each dot is a single prediction (row) from the dataset.
            "The x-axis is the value of the feature (from the X matrix, stored in explanation.data).
            "The y-axis is the SHAP value for that feature (stored in explanation.values), which represents how much knowing that feature’s value changes the output of the model for that sample’s prediction.
            "The light grey area at the bottom of the plot is a histogram showing the distribution of data values.

            The dependence scatter plot graph is attached as an image\n
            """
        prompt_text += f"""
        Description of the dataset that the model was trained on: {self.dataset_context}
        Machine learning model task description: {self.model_task_context}
        Feature name: {feature_name}
        Feature description: {self.features_descriptions[feature_name]}
        Features type:  {self.X_dataset[feature_name].dtype}
        \n\n
        """


        return [
            {
                "type": "text",
                "text": prompt_text
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image_graph}",
                    "detail": "low"
                }
            }
        ]


    def prompt_describe_graph_cot(self, base64_image_graph, feature_name, num_sentences=7):
        """Use chain-of-thought reasoning to elicit a description of a graph in at most {num_sentences} sentences.

        Returns:
            Messages in OpenAI format.
        """
        system_prompt = ( "You are helping users understand an ML model's prediction. "
        "Given an explanation and information about the model, "
        "convert the explanation into a human-readable narrative."
        "You are an expert data scientist."
        "You answer all questions to the best of your ability, relying on the graphs provided by the user, any other information you are given, and your knowledge about the real world.")

        chain = "Solve the tasks one by one:\\n"
        chain += "task1: Please describe the general pattern of the graph."
        chain += "task2: Please study the graph carefully and highlight any regions you may find surprising or counterintuitive. You may also suggest an explanation for why this behavior is surprising, and what may have caused it."
        chain += f"task3: Please provide a brief summary, at most {num_sentences} sentence description of the graph. Be sure to include any important surprising patterns in the description."
        messages = [
            {"role": "system", "content": system_prompt + chain} ,
            {"role": "user", "content": self.prompt_describe_graph(base64_image_graph, feature_name)},
        ]
        return messages

    def get_improved_metadata(self):

        prompt_system = """You are helping users understand an ML model features.
        Given table name and features information, provide easier to understand names and detailed descriptions for each feature in a language even non domain exper can understand
        Follow the following format
        Description of the dataset that the model was trained on: General information about the dataset
        Machine learning model task description: what the model predicts
        Features information: Information about the features
        Features information Format: format the features information is given in
        New description: detailed descriptions for each feature
        New name: concise names for  each feature."""

        features_information = ",".join([f"({name}, {desc})" for name, desc in self.features_descriptions.items()])
        prompt_user = f"""Description of the dataset that the model was trained on: {self.dataset_context}
        Machine learning model task description: {self.model_task_context}
        Features information: {features_information}
        Features information format:  feature information in (feature name, feature current description) format"""

        messages = [{"role": "system", "content": prompt_system},
                    {"role": "user", "content": prompt_user}]

        class ColumnsDescription(BaseModel):
            name: str
            new_name: str
            new_description: str

        class ColumnsDescriptions(BaseModel):
            descriptions: list[ColumnsDescription]

        completion = self.llm_client.beta.chat.completions.parse(
            model=self.llm_model,
            messages=messages,
            response_format=ColumnsDescriptions,
        )

        columnsDescriptions = completion.choices[0].message
        return columnsDescriptions
