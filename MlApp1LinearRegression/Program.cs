using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using Microsoft.ML.Trainers;
using System;
using Microsoft.ML.Models;

namespace MlApp1LinearRegression
{
    class Program
    {
        static void Main(string[] args)
        {
            //https://www.youtube.com/watch?v=8gVhJKszzzI
            #region
            //Setup the algorithm
            //LearningPipeline from Microsoft.ML;
            //TextLoader from Microsoft.ML.Data
            //ColumnConcatenator from Microsoft.ML.Transforms;
            //GeneralizedAdditiveModelRegressor from Microsoft.ML.Trainers;

            //TextLoader connects to CSV file
            //separator (delimiter)  needs single quotes for some reason
            //Create from is generic so can take any generic derrived type
            //ColumnConCatenator concatates your input data into one column, calls it 'features', and pass in the column to this - we only have one - YearsExperience
            //error I got - features needed to be capital first as per namign convention - threw exception when not
            #endregion
            var pipeline = new LearningPipeline();
            pipeline.Add(new TextLoader("SalaryData.csv").CreateFrom<SalaryData>(useHeader: true, separator: ','));
            pipeline.Add(new ColumnConcatenator("Features", "YearsExperience"));
            pipeline.Add(new GeneralizedAdditiveModelRegressor());

            #region
            //call train method on the pipelien will train on the data it has
            //object is generic derrived also
            //two arguments - one is data input, next is dataoutput
            //requires the data input in one column
            #endregion
            var model = pipeline.Train<SalaryData, SalaryPrediction>();
            #region
            //testing the effectiveness of the model
            //RegressionEvaluator from Microsoft.ML.Models
            //map to a newfile called SalaryData-test.csv which holds the test data set (not the  cross validation data set)
            //we can use this to stop overfitting / other errors
            //RegressionEvaluator - call Evaluate method on this and it will give you... metrics to checkl teh effectiveness of the model
            //the above method requires the model and test data passing in to check how effective it has been
            //the best evaluator metrics are the root mean squared and the r squared values ) 
            //RMS gives the average actual value our model was out by (it gives on average $4417)
            //The R^2 vlaue is teh value I have been seeign in the course - how far the actual values in the test data are from the relative model data - the total error - closer to 1, teh better your model is... there are some problems with this value ( I am guessin potential overfitting closer to 1)
            #endregion
            var testData = new TextLoader("SalaryData-test.csv").CreateFrom<SalaryData>(useHeader: true, separator: ',');
            var evaluator = new RegressionEvaluator();
            var metrics = evaluator.Evaluate(model, testData);

            Console.WriteLine($"RMS - {metrics.Rms}");
            Console.WriteLine($"R^2 - {metrics.RSquared}");

            #region
            //predict on unseen data
            //call predict method on the model, pass in a new SalaryData model - set the Feature you want to predict on
            #endregion
            var prediction = model.Predict(new SalaryData { YearsExperience = 7 });
            Console.WriteLine($"Predicted Salary for 7 = {prediction.PredictedSalary}");


            Console.ReadLine();

        }
    }
}
