using Microsoft.ML.Runtime.Api;
using System;
using System.Collections.Generic;
using System.Text;

namespace MlApp1LinearRegression
{
    class SalaryData
    {
        #region
        //Column attributes
        //need ML import for these to take
        //these tell ML.Net which position each column is in so we can map the data to each field
        #endregion
        [Column("0")]
        public float YearsExperience { get; set; }
        [Column("1", name: "Label")]
        public float Salary { get; set; }
    }
}
