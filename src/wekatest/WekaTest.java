/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package wekatest;

import java.util.Scanner;
import weka.classifiers.trees.RandomForest;
import weka.core.DenseInstance;
import weka.core.Instance;

import weka.core.Instances;
import weka.core.converters.*;
import weka.core.converters.ConverterUtils.DataSource;

/**
 *
 * @author monirozzamanroni
 */
public class WekaTest {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws Exception {
        
                DataSource source1 = new DataSource("heart.arff");
	        Instances train = source1.getDataSet();
	        // setting class attribute if the data format does not provide this information
	        // For example, the XRFF format saves the class attribute information as well
	        if (train.classIndex() == -1)
	            train.setClassIndex(train.numAttributes() - 1);

	        DataSource source2 = new DataSource("test.arff");
	        Instances test = source2.getDataSet();
               
	        // setting class attribute if the data format does not provide this information
	        // For example, the XRFF format saves the class attribute information as well
	        if (test.classIndex() == -1)
	            test.setClassIndex(train.numAttributes() - 1);

	        // model

	        RandomForest randomForest = new RandomForest();
	        randomForest.buildClassifier(train);
                
//                //input from user
//                String outlook,temperature,humidity,windy,requestData;
//                Scanner s = new Scanner(System.in);
//                System.out.println("Please enter marks:");
//                outlook  = s.nextLine();
//                temperature  = s.nextLine();
//                humidity  = s.nextLine();
//                windy  = s.nextLine();
//                System.out.println(test.instance(0));
                
               double[] instanceValue1 = new double[test.numAttributes()];
                instanceValue1[0] = 2;
                instanceValue1[1] = 71;
                instanceValue1[2] = 91;
                instanceValue1[3] = 0;
              
                test.add(new DenseInstance(1.0, instanceValue1));

	        double label = randomForest.classifyInstance(test.instance(0));
	        test.instance(0).setClassValue(label);
       
	        System.out.println(test.instance(0).stringValue(4));
    }
    
}
