package jluccst.keras2java;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.PMML;
import org.jpmml.model.PMMLUtil;
import org.jpmml.evaluator.Evaluator;
import org.jpmml.evaluator.FieldValue;
import org.jpmml.evaluator.InputField;
import org.jpmml.evaluator.ModelEvaluatorFactory;
import org.jpmml.evaluator.ProbabilityDistribution;
import org.jpmml.evaluator.TargetField;
import org.jpmml.evaluator.ValueMap;
import org.jpmml.evaluator.support_vector_machine.VoteDistribution;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.xml.sax.SAXException;
import javax.xml.bind.JAXBException;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class KerasTest {

	private MultiLayerNetwork keras_model;
	private Evaluator modelEvaluator;

	public void loadModle(String name) {
		try {
			keras_model = KerasModelImport.importKerasSequentialModelAndWeights(name);
			System.out.println(keras_model.summary());
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (InvalidKerasConfigurationException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (UnsupportedKerasConfigurationException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	public void loadModle(String json, String weight) {
		try {
			keras_model = KerasModelImport.importKerasSequentialModelAndWeights(json, weight);
			System.out.println(keras_model.summary());
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (InvalidKerasConfigurationException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (UnsupportedKerasConfigurationException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	public INDArray predict(INDArray input) {
		// get output of layer 'dense_1'
		List<INDArray> result = keras_model.feedForwardToLayer(10, input, false);
		return result.get(9);
	}

	public INDArray readDatafromDB(int id) {
		int channel_count = 56;
		int sample_count = 500;
		INDArray input = Nd4j.zeros(1, channel_count, sample_count);
		Connection con;
		String driver = "com.mysql.jdbc.Driver";
		String url = "jdbc:mysql://localhost:3306/yf15";
		String user = "root";
		String password = "mysql";
		try {
			Class.forName(driver);
			con = DriverManager.getConnection(url, user, password);
			if (!con.isClosed())
				System.out.println("Succeeded connecting to the Database!");
			Statement statement = con.createStatement();
			String sql = "SELECT WellDep, HangDownDeep, DepthOfTheDrillBit, DrillDownDeep, DrillingTime, BitPressure, "
					+ "HangingLoad, RotationRate, Torque, KellyDown, HookPosition, HookSpeed, StandpipePressureLog, "
					+ "CasingPressure, PumpStroke1, PumpStroke2, PumpStroke3, TotalPoolSize, LayTime, MudSpill, "
					+ "InletFlowlog, OutletFlowlog, InletDensitylog, OutletDensitylog, EntranceTempreture, ExitTempreture, "
					+ "TotalHydrocarbon, H2S, C1, C2, PWDHangDownDeep, PWD_hkyl, PWDWellDeviation, PWDLocation,"
					+ "UpturnDepth, OperatorSchema, StandpipePressure, MeasurementOfBackPressure, OutletFlow, OutletDensity, "
					+ "BackPressurePumpFlow, LoopBackPressure, AdditionalBackPressure, InletFlow, FixDepth, FixPointDownDeep, "
					+ "FixPointPressure, WellMouthAdjustment, FixPointPressureLoss, FixPointECD, DrillECD, DrillStringPressureDrop, "
					+ "DrillBitPressureDrop, EnvironmentalControlPressureLoss, TargetBackPressure, HydrostaticPressure"
					+ " from oil_2 order by Time desc limit 500;";
			ResultSet rs = statement.executeQuery(sql);
			int i = 0;
			while (rs.next()) {
				for (int j = 1; j <= channel_count; j++) {
					double val = rs.getDouble(j);
					input.putScalar(0, j - 1, i, val);
				}
				i++;
			}
			rs.close();
			con.close();
		} catch (ClassNotFoundException e) {
			// 数据库驱动类异常处理
			System.out.println("Sorry,can`t find the Driver!");
			e.printStackTrace();
		} catch (SQLException e) {
			// 数据库连接失败异常处理
			e.printStackTrace();
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		return input;
	}

	public void loadSVMModel(String pmmlFileName) {
		PMML pmml = null;

		try {
			if (pmmlFileName != null) {
				InputStream is = new FileInputStream(pmmlFileName);
				pmml = PMMLUtil.unmarshal(is);
				try {
					is.close();
				} catch (IOException e) {
					System.out.println("InputStream close error!");
				}

				ModelEvaluatorFactory modelEvaluatorFactory = ModelEvaluatorFactory.newInstance();

				this.modelEvaluator = (Evaluator) modelEvaluatorFactory.newModelEvaluator(pmml);
				modelEvaluator.verify();
				System.out.println("加载模型成功！");
			}
		} catch (SAXException e) {
			e.printStackTrace();
		} catch (JAXBException e) {
			e.printStackTrace();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}

	}

	public Map<String, Double> convertTo(INDArray outer) {
		
		Map<String, Double>  map2=new HashMap<String, Double>();
		for(int i = 0; i<=outer.columns()-1;i++) {
			map2.put("x"+String.valueOf(i+1),outer.getDouble(i));
		}
		return map2;

	}
	
public Map<String, Double> convertTo2750(INDArray outer) {
		
		Map<String, Double>  map2=new HashMap<String, Double>();
		for(int i = 0; i<=outer.size(1)-1;i++) {
			for(int j = 0; j<=outer.size(2)-1;j++) {
				map2.put("x"+String.valueOf(i*outer.size(2)+j+1),outer.getDouble(0,i,j));
			}
			
		}
		return map2;

	}

	public String svm_predict(Map<String, Double> kxmap) {
		String ca = "-1";

		List<InputField> inputFields = modelEvaluator.getInputFields();

		Map<FieldName, FieldValue> arguments = new LinkedHashMap<FieldName, FieldValue>();
		for (InputField inputField : inputFields) {
			FieldName inputFieldName = inputField.getName();
			Object rawValue = kxmap.get(inputFieldName.getValue());
			FieldValue inputFieldValue = inputField.prepare(rawValue);
			arguments.put(inputFieldName, inputFieldValue);
		}
		Map<FieldName, ?> results = modelEvaluator.evaluate(arguments);
		List<TargetField> targetFields = modelEvaluator.getTargetFields();
		for (TargetField targetField : targetFields) {
			FieldName targetFieldName = targetField.getName();
			VoteDistribution targetFieldValue = (VoteDistribution) results.get(targetFieldName);
			System.out.println("target: " + targetFieldName.getValue() + " value: " + targetFieldValue);
			Set<String> categories = targetFieldValue.getCategories();
			Double max = 0.0;
			for (String category : categories) {
				if (targetFieldValue.getProbability(category) > max) {
					max = targetFieldValue.getProbability(category);
					ca = category;
				}
			}

		}
		return ca;
	}

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		KerasTest kt = new KerasTest();
		INDArray input = kt.readDatafromDB(0);
		System.out.println(input.shapeInfoToString());
		kt.loadModle("classify1CNN.h5");
//		kt.loadModle("model.json", "model.json.h5");
		INDArray outer = kt.predict(input);
		System.out.println(outer.shapeInfoToString());
		kt.loadSVMModel("svm.pmml");
		Map<String, Double>  map2 = kt.convertTo2750(outer);
		System.out.println(kt.svm_predict(map2));

	}

}
