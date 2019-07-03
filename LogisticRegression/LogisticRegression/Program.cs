using System;
namespace LogisticRegression {
	class Program {
		static void Main(string[] args) {
			Console.WriteLine("\nBegin Logistic Regression Binary Classification demo");
			Console.WriteLine("Goal is to predict death (0 = false, 1 = true)");
			double[][] data = new double[150][];
			data = LoadData("iris.csv", 150, 5, ' ');
		/*	data[0] = new double[] { 0, 85, 85, 0,0 };
			data[1] = new double[] { 0, 80, 90, 1,0 };
			data[2] = new double[] { 1, 83, 86, 0,1 };
			data[3] = new double[] { 2, 70, 96, 0,1 };
			data[4] = new double[] {2, 68, 80, 0,1 };
			data[5] = new double[] { 2, 65, 70, 1 ,0};
			data[6] = new double[] { 1, 64, 65, 1,1 };
			data[7] = new double[] { 0, 72, 95, 0,0 };
			data[8] = new double[] { 0, 69, 70, 0,1 };
			data[9] = new double[] { 2, 75, 80, 0,1 };
			data[10] = new double[] { 0,75, 70, 1,1 };
			data[11] = new double[] { 1, 72, 90, 1,1 };
			data[12] = new double[] { 1, 81, 75, 0,1 };
			data[13] = new double[] { 2, 71, 91, 1,0 };*/
			Console.WriteLine("\nRaw data: \n");
			Console.WriteLine(" Age Sex Kidney Died");
			Console.WriteLine("=======================================");
			//ShowData(data, 5, 2, true);
			Console.WriteLine("Normalizing age and kidney data");
			int[] columns = new int[] { 0,1,2,3};
			double[][] means = Normalize(data, columns); // normalize, save means and stdDevs
			Console.WriteLine("Done");
			Console.WriteLine("\nNormalized data: \n");
			//ShowData(data, 5, 2, true);
			Console.WriteLine("Creating train (80%) and test (20%) matrices");
			double[][] trainData;
			double[][] testData;
			MakeTrainTest(data, 0, out trainData, out testData);
			Console.WriteLine("Done");
			Console.WriteLine("\nNormalized training data: \n");
			//ShowData(trainData, 3, 2, true); //Console.WriteLine("\nFirst 3 rows and last row of normalized test data: \n");
			//ShowData(testData, 3, 2, true);
			int numFeatures = 4; // number of x-values (age, sex, kidney)
			Console.WriteLine("Creating LR binary classifier");
			LogisticClassifier lc = new LogisticClassifier(numFeatures);
			int maxEpochs = 999; // gives a representative demo
			Console.WriteLine("Setting maxEpochs = " + maxEpochs);
			Console.WriteLine("Starting training using simplex optimization");
			double[] bestWeights = lc.Train(trainData, maxEpochs, 33); // 33 = 'nice' demo
			Console.WriteLine("Training complete");
			Console.WriteLine("\nBest weights found:");
			ShowVector(bestWeights, 5, true);
			double trainAccuracy = lc.Accuracy(trainData, bestWeights);
			Console.WriteLine("Prediction accuracy on training data = " + trainAccuracy.ToString("F4"));
			double testAccuracy = lc.Accuracy(testData, bestWeights);
			Console.WriteLine("Prediction accuracy on test data = " + testAccuracy.ToString("F4")); //double[][] unknown = new double[1][]; //unknown[0] = new double[] { 58.0, -1.0, 7.00 }; //Normalize(unknown, columns, means); //int died = lc.ComputeDependent(unknown[0], bestWeights); //Console.WriteLine("Died = " + died);
			Console.WriteLine("\nEnd LR binary classification demo\n");
			Console.ReadLine();
		} // Main
		static double[][] LoadData(string dataFile, int numRows, int numCols, char delimit) {
			System.IO.FileStream ifs = new System.IO.FileStream(dataFile, System.IO.FileMode.Open); 
			System.IO.StreamReader sr = new System.IO.StreamReader(ifs); 
			string line = "";
			string[] tokens = null; 
			int i = 0; 
			double[][] result = new double[numRows][];
			while ((line = sr.ReadLine()) != null) { 
				result[i] = new double[numCols]; 
				tokens = line.Split(delimit); 
				for (int j = 0; j < numCols; ++j) 
					result[i][j] = double.Parse(tokens[j]); 
				++i; 
			} 
			sr.Close(); 
			ifs.Close(); 
			return result; 
		}
		static double[][] Normalize(double[][] rawData, int[] columns) { // return means and sdtDevs of all columns for later use
			int numRows = rawData.Length;
			int numCols = rawData[0].Length;
			double[][] result = new double[2][]; // [0] = mean, [1] = stdDev
			for (int i = 0; i < 2; ++i)
				result[i] = new double[numCols];
			for (int c = 0; c < numCols; ++c) { // means of all cols
				double sum = 0.0;
				for (int r = 0; r < numRows; ++r)
					sum += rawData[r][c];
				double mean = sum / numRows;
				result[0][c] = mean; // save
				// stdDevs of all cols
				double sumSquares = 0.0;
				for (int r = 0; r < numRows; ++r)
					sumSquares += (rawData[r][c] - mean) * (rawData[r][c] - mean);
				double stdDev = Math.Sqrt(sumSquares / numRows);
				result[1][c] = stdDev;//her sütun için standart sapması hesaplanıyor
			} // normalize
			for (int c = 0; c < columns.Length; ++c)	 {
				int j = columns[c]; // column to normalize
				double mean = result[0][j]; // mean of the col
				double stdDev = result[1][j];
				for (int i = 0; i < numRows; ++i) rawData[i][j] = (rawData[i][j] - mean) / stdDev; }
			return result;
		}
		static void Normalize(double[][] rawData, int[] columns, double[][] means) { // normalize columns using supplied means and standard devs
			int numRows = rawData.Length;
			for (int c = 0; c < columns.Length; ++c) // each specified col
			{
				int j = columns[c]; // column to normalize
				double mean = means[0][j];
				double stdDev = means[1][j];
				for (int i = 0; i < numRows; ++i) // each row
					rawData[i][j] = (rawData[i][j] - mean) / stdDev;
			}
		}
		static void MakeTrainTest(double[][] allData, int seed, out double[][] trainData, out double[][] testData) {
			Random rnd = new Random(seed);
			int totRows = allData.Length;
			int numTrainRows = (int)(totRows * 0.80); // 80% hard-coded
			int numTestRows = totRows - numTrainRows;
			trainData = new double[numTrainRows][];
			testData = new double[numTestRows][];
			double[][] copy = new double[allData.Length][]; // ref copy of all data
			for (int i = 0; i < copy.Length; ++i) copy[i] = allData[i];
			for (int i = 0; i < copy.Length; ++i) // scramble order
			{ int r = rnd.Next(i, copy.Length); // use Fisher-Yates
				double[] tmp = copy[r]; copy[r] = copy[i]; copy[i] = tmp;
			}
			for (int i = 0; i < numTrainRows; ++i)
				trainData[i] = copy[i];
			for (int i = 0; i < numTestRows; ++i)
				testData[i] = copy[i + numTrainRows];
		} // MakeTrainTest
		static void ShowData(double[][] data, int numRows, int decimals, bool indices) {
			for (int i = 0; i < 24; ++i) {
				if (indices == true) Console.Write("[" + i.ToString().PadLeft(2) + "] ");
				for (int j = 0; j < data[i].Length; ++j) {
					double v = data[i][j]; if (v >= 0.0)
						Console.Write(" "); // '+'
					Console.Write(v.ToString("F" + decimals) + " ");
				} Console.WriteLine("");
			} Console.WriteLine(". . .");
			int lastRow = data.Length - 1;
			if (indices == true)
				Console.Write("[" + lastRow.ToString().PadLeft(2) + "] ");
			for (int j = 0; j < data[lastRow].Length; ++j) {
				double v = data[lastRow][j];
				if (v >= 0.0) Console.Write(" "); // '+'
				Console.Write(v.ToString("F" + decimals) + " ");
			} Console.WriteLine("\n");
		}
		static void ShowVector(double[] vector, int decimals, bool newLine) {
			for (int i = 0; i < vector.Length; ++i)
				Console.Write(vector[i].ToString("F" + decimals) + " ");
			Console.WriteLine("");
			if (newLine == true)
				Console.WriteLine("");
		}
	} // Program
	public class LogisticClassifier {
		private int numFeatures; // number of independent variables aka features
		private double[] weights; // b0 = constant
		private Random rnd;
		public LogisticClassifier(int numFeatures) {
			this.numFeatures = numFeatures; // number of features/predictors
			this.weights = new double[numFeatures + 1]; // [0] = b0 constant
		}
		public double[] Train(double[][] trainData, int maxEpochs, int seed) {
			// sort 3 solutions (small error to large)
			// compute centroid
			// if expanded is better than worst replace
			// else if reflected is better than worst, replace
			// else if contracted is better than worst, replace
			// else if random is better than worst, replace
			// else shrink
			this.rnd = new Random(seed); // so we can implement restart if wanted
			Solution[] solutions = new Solution[3]; // best, worst, other // initialize to random values
			for (int i = 0; i < 3; ++i) {
				solutions[i] = new Solution(numFeatures);
				solutions[i].weights = RandomSolutionWts();
				solutions[i].error = Error(trainData, solutions[i].weights);
			}
			int best = 0; // for solutions[idx].error
			int other = 1;
			int worst = 2;
			int epoch = 0;
			while (epoch < maxEpochs) {
				++epoch;
				Array.Sort(solutions); // [0] = best, [1] = other, [2] = worst
				double[] bestWts = solutions[0].weights; // convenience only
				double[] otherWts = solutions[1].weights;
				double[] worstWts = solutions[2].weights;
				double[] centroidWts = CentroidWts(otherWts, bestWts); // an average
				double[] expandedWts = ExpandedWts(centroidWts, worstWts);
				double expandedError = Error(trainData, expandedWts);
				if (expandedError < solutions[worst].error) // expanded better than worst?
				{
					Array.Copy(expandedWts, worstWts, numFeatures + 1); // replace worst
					solutions[worst].error = expandedError; continue;
				}
				double[] reflectedWts = ReflectedWts(centroidWts, worstWts);
				double reflectedError = Error(trainData, reflectedWts);
				if (reflectedError < solutions[worst].error) // relected better than worst?
				{ Array.Copy(reflectedWts, worstWts, numFeatures + 1); solutions[worst].error = reflectedError; continue;
				}
				double[] contractedWts = ContractedWts(centroidWts, worstWts);
				double contractedError = Error(trainData, contractedWts);
				if (contractedError < solutions[worst].error) // contracted better than worst?
				{
					Array.Copy(contractedWts, worstWts, numFeatures + 1);
					solutions[worst].error = contractedError;
					continue;
				}
				double[] randomSolWts = RandomSolutionWts();
				double randomSolError = Error(trainData, randomSolWts);
				if (randomSolError < solutions[worst].error)
				{
					Array.Copy(randomSolWts, worstWts, numFeatures + 1);
					solutions[worst].error = randomSolError; continue;
				} // couldn't find a replacement for worst so shrink // worst -> towards best
				for (int j = 0; j < numFeatures + 1; ++j)
					worstWts[j] = (worstWts[j] + bestWts[j]) / 2.0;
				solutions[worst].error = Error(trainData, worstWts); // 'other' -> towards best
				for (int j = 0; j < numFeatures + 1; ++j)
					otherWts[j] = (otherWts[j] + bestWts[j]) / 2.0;
				solutions[other].error = Error(trainData, otherWts);
			} // while // copy best weights found, return by reference
			Array.Copy(solutions[best].weights, this.weights, this.numFeatures + 1);
			return this.weights;
		}

		private double[] CentroidWts(double[] otherWts, double[] bestWts) {
			double[] result = new double[this.numFeatures + 1];
			for (int i = 0; i < result.Length; ++i) result[i] = (otherWts[i] + bestWts[i]) / 2.0;
			return result;
		}
		private double[] ExpandedWts(double[] centroidWts, double[] worstWts) {
			double gamma = 2.0; // how far from centroid
			double[] result = new double[this.numFeatures + 1];
			for (int i = 0; i < result.Length; ++i)
				result[i] = centroidWts[i] + (gamma * (centroidWts[i] - worstWts[i]));
			return result;
		}
		private double[] ReflectedWts(double[] centroidWts, double[] worstWts) {
			double alpha = 1.0; // how far from centroid
			double[] result = new double[this.numFeatures + 1];
			for (int i = 0; i < result.Length; ++i)
				result[i] = centroidWts[i] + (alpha * (centroidWts[i] - worstWts[i]));
			return result;
		}
		private double[] ContractedWts(double[] centroidWts, double[] worstWts) {
			double rho = -0.5;
			double[] result = new double[this.numFeatures + 1];
			for (int i = 0; i < result.Length; ++i)
				result[i] = centroidWts[i] + (rho * (centroidWts[i] - worstWts[i]));
			return result;
		}
		private double[] RandomSolutionWts() {
			double[] result = new double[this.numFeatures + 1];
			double lo = -10.0;
			double hi = 10.0;
			for (int i = 0; i < result.Length; ++i)
				result[i] = (hi - lo) * rnd.NextDouble() + lo;
			return result;
		}
		private double Error(double[][] trainData, double[] weights) { // mean squared error using supplied weights
			int yIndex = trainData[0].Length - 1; // y-value (0/1) is last column
			double sumSquaredError = 0.0;
			for (int i = 0; i < trainData.Length; ++i) // each data
			{
				double computed = ComputeOutput(trainData[i], weights);
				double desired = trainData[i][yIndex]; // ex: 0.0 or 1.0
				sumSquaredError += (computed - desired) * (computed - desired);
			}
			return sumSquaredError / trainData.Length;
		} public double ComputeOutput(double[] dataItem, double[] weights) {
			double z = 0.0; z += weights[0]; // the b0 constant
			for (int i = 0; i < weights.Length - 1; ++i) // data might include Y
				z += (weights[i + 1] * dataItem[i]); // skip first weight
			return 1.0 / (1.0 + Math.Exp(-z));
		}
		public int ComputeDependent(double[] dataItem, double[] weights) {
			double sum = ComputeOutput(dataItem, weights);
			if (sum <= 0.5) return 0; else return 1;
		}
		public double Accuracy(double[][] trainData, double[] weights) {
			int numCorrect = 0;
			int numWrong = 0;
			int yIndex = trainData[0].Length - 1;
			for (int i = 0; i < trainData.Length; ++i) {
				double computed = ComputeDependent(trainData[i], weights); // implicit cast
				double desired = trainData[i][yIndex]; // 0.0 or 1.0
				if (computed == desired) // risky?
					++numCorrect;
				else ++numWrong; //double closeness = 0.00000001;
				//if (Math.Abs(computed - desired) < closeness)
				// ++numCorrect; //else // ++numWrong;
			} return (numCorrect * 1.0) / (numWrong + numCorrect);
		}
		private class Solution : IComparable<Solution> {
			public double[] weights; // a potential solution
			public double error; // MSE of weights
			public Solution(int numFeatures) {
				this.weights = new double[numFeatures + 1]; // problem dim + constant
				this.error = 0.0;
			} public int CompareTo(Solution other) // low-to-high error
			{
				if (this.error < other.error)
					return -1;
				else if (this.error > other.error)
					return 1; else return 0;
			}
		}
	}
}