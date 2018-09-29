using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Runtime.Api;

namespace MlApp1LinearRegression
{
    class SalaryPrediction
    {
        [ColumnName("Score")]
        public float PredictedSalary{ get; set; }
    }
}
