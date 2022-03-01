package cloud.workflowScheduling.idea.classificationScheduling;

import java.io.*;
import java.math.*;
import java.util.*;

import org.apache.commons.math3.stat.*;


import cloud.workflowScheduling.idea.subDMethods.*;
import cloud.workflowScheduling.methods.*;
import cloud.workflowScheduling.setting.*;

/*
 * 算法性能比较
 * Please download the DAX workflow archive from 
 * https://download.pegasus.isi.edu/misc/SyntheticWorkflows.tar.gz, 
 * unzip it and keep the DAX workflows in an appropriate position before running.
 */
public class Main {
	// because in Java float numbers can not be precisely stored, a very small
	// number E is added before testing whether deadline is met
	public static final double E = 0.0000001;

	private static double DF_START = 1, DF_INCR = 1, DF_END = 10;
	
	private static final int REPEATED_TIMES = 10;
	private static final int FILE_INDEX_MAX = 200; //每类生成了100个xml
	private static final int[] SIZES = {30, 50, 100, 1000}; //30, 50, 100, 1000
	private static int WorkflowNum = 250; //每种size的工作流个数

	//算法new之后一直在用，成员变量的值会保存上一次的执行，所以进入算法后，先初始化成员变量
	//new ClassifySchedule(), new Method1uRank(), new Method2ProLiS(), new Method6(), new Method3(), new ICPCP(), new ProLiS(1.5), new PSO()
	private static final Scheduler[] METHODS = {new ClassifySchedule(), new Method1uRank(), new Method2ProLiS(), new Method6(), new Method3(), new ICPCP(), new ProLiS(1.5), new PSO()}; 
	//"CyberShake","Epigenomics","Inspiral", "Montage", "Sipht"
	private static final String[] WORKFLOWS = {"CyberShake","Epigenomics","Inspiral", "Montage", "Sipht"};

	static final String WORKFLOW_LOCATION = "E:\\0Work\\1Ideas\\workflowClassificationAndScheduling\\workflowScheduling\\workflowGenerator\\WorkflowGenerator\\bharathi\\generatedWorkflows";
	static final String OUTPUT_LOCATION = ".\\result1";
	static final String ExcelFilePath = ".\\result1\\solution.xls";
	public static ExcelManage em;
	public static String sheetName;
	public static boolean isPrintExcel = false; // true false

	public static HashMap<String, HashMap<Integer, Double>> type2subD = new HashMap<String, HashMap<Integer, Double>>();
	public static Map<String, Integer> workflow2lable = new LinkedHashMap<String, Integer>();
	public static List<Double> deadlines = new ArrayList<Double>();
	public static List<Double> deadlineFactors = new ArrayList<Double>();
	public static HashMap<String, String> classifyResult = new HashMap<String, String>();
	//每一种工作流的每一种size对应的minCost和maxCost
	public static HashMap<String, List<Double>> minMaxCost = new LinkedHashMap<String, List<Double>>();
	
	static boolean isCalMinMaxCost = false; //是否计算MinMaxCost：true计算，false从文件读取计算好的
//	public static boolean isCallClassifyMethod = true; // 是否调用分类算法
	public static boolean onlyCalCostOfFeasible = true; //是否仅仅计算可行解的cost，之前跑的都是false
	
	public static BufferedWriter bwUpdate = null;
	
	public static void main(String[] args) throws Exception {
		if(isCalMinMaxCost)
			calMinMaxCost();
		else {
		String minMaxFile = "minMaxCost2.txt";	
		readMinMaxCost(minMaxFile);
		bwUpdate = new BufferedWriter(new FileWriter(OUTPUT_LOCATION + "\\update_minMaxCost.txt"));
		bwUpdate.close();
		
		//计算每种类型工作流的每种size在不同方法下的结果，例如Cyber在size30，不同方法在不同deadline上的结果
//		//WorkflowNum = FILE_INDEX_MAX;
//		List<List<String>> workflowList = new ArrayList<List<String>>();
//		for(int si = 0; si < SIZES.length; si++) {
//			int size = SIZES[si];
//			List<String> w = new ArrayList<String>();
//			for(int fi = 0;fi<FILE_INDEX_MAX;fi++) {
//				w.add(WORKFLOWS[0] +"_"+ size + "." + fi + ".xml");
//			}
//			workflowList.add(w);
//		}
		
		//随机生成每种size的工作流
//		selectWorkflows(FILE_INDEX_MAX, WorkflowNum); 
		//读取随机生成的工作流
		List<List<String>> workflowList = new ArrayList<List<String>>();
		for(int si = 0; si < SIZES.length; si++) {
			List<String> w = new ArrayList<String>();
			String fileName = OUTPUT_LOCATION + "\\" + "selectedWorkflows_" + SIZES[si] + ".txt";
			w = getWorkflowListFromFile(fileName);
			workflowList.add(w);
		}
		
		//读取分类结果
		String crFile = "E:\\0Work\\1Ideas\\workflowClassificationAndScheduling\\DAGclassificationWithTT-test\\"
							+ "新result-wen+划分+步长\\epoch199_classificationResult.txt";	
//		String crFile = "E:\\0Work\\1Ideas\\workflowClassificationAndScheduling\\workflowScheduling\\"
//							+ "workflow2subDLable\\result\\repeat\\workflow2label2_200.txt";
		getClassifyResultFromFile(crFile);
//		
		if (isPrintExcel)
			ExcelManage.clearExecl(ExcelFilePath);
		int deadlineNum = (int) ((DF_END - DF_START) / DF_INCR + 1);
		for (int si = 0; si < SIZES.length; si++) {
			int size = SIZES[si];
			double[][][] successResult = new double[deadlineNum][METHODS.length][REPEATED_TIMES*WorkflowNum];
			double[][][] NCResult = new double[deadlineNum][METHODS.length][REPEATED_TIMES*WorkflowNum];
			double[] refValues = new double[4]; // store cost and time of fastSchedule and cheapSchedule
			
			int ith = 0;
			for(String workflowFile : workflowList.get(si)) {
				String file = WORKFLOW_LOCATION + "\\" + workflowFile;
				for (int di = 0; di <= (DF_END - DF_START) / DF_INCR; di++) { // deadline index
					for (int timeI = 0; timeI < REPEATED_TIMES; timeI++) {
						test(file, di, timeI, ith, successResult, NCResult, refValues);
					}
				}
				ith++;
			}	
			BufferedWriter bw = new BufferedWriter(new FileWriter(OUTPUT_LOCATION + "\\size" + size + ".txt"));
			bw.write("used methods: ");
			for (Scheduler s : METHODS)
				bw.write(s.getClass().getSimpleName() + "\t");
			bw.write("\r\n\r\n");
			printTo(bw, successResult, "success ratio");
			if(onlyCalCostOfFeasible)
				printCostOfFeasibleTo(bw, successResult, NCResult, "normalized cost"); //计算REPEATED_TIMES中某些次可行解的平均cost
			else
				printTo(bw, NCResult, "normalized cost");

			bw.write("reference values (CF, MF, CC, MC)\r\n");
			double divider = SIZES.length * FILE_INDEX_MAX * deadlineNum;
			for (double refValue : refValues)
				bw.write(refValue / divider + "\t");
			bw.close();
		}
		}
	}
	
	//计算minCost和maxCost
	private static void calMinMaxCost() throws IOException {
		int deadlineNum = (int) ((DF_END - DF_START) / DF_INCR + 1);
		
		BufferedWriter bw = new BufferedWriter(new FileWriter(OUTPUT_LOCATION + "\\minMaxCost.txt", true));
		bw.close();
		for (String workflow : WORKFLOWS) {
			for (int si = 0; si < SIZES.length; si++) {
				int size = SIZES[si];
				Main.minMaxCost.put(workflow +"_"+ size, new ArrayList<Double>());
				Main.minMaxCost.get(workflow +"_"+ size).add(Double.MAX_VALUE);
				Main.minMaxCost.get(workflow +"_"+ size).add(Double.MIN_VALUE);
				bw = new BufferedWriter(new FileWriter(OUTPUT_LOCATION + "\\minMaxCost.txt", true));
				for(int fi = 0;fi<FILE_INDEX_MAX;fi++){
					String file = WORKFLOW_LOCATION + "\\" + workflow +"_"+ size + "." + fi + ".xml";
					for (int di = 0; di <= (DF_END - DF_START) / DF_INCR; di++) { // deadline index
						for (int timeI = 0; timeI < REPEATED_TIMES; timeI++) {
							testForMinMaxCost(file, di);
						}
					}
				}
				
				bw.write(workflow +"_"+ size+", ");
				for(Double in : Main.minMaxCost.get(workflow +"_"+ size)) {
					bw.write(in.doubleValue() + ", ");
				}
				bw.write("\r\n");
				bw.close();
			}
		}
			
//		BufferedWriter bw = new BufferedWriter(new FileWriter(OUTPUT_LOCATION + "\\minMaxCost.txt"));
//		Set<Map.Entry<String,List<Double>>> entrySet = Main.minMaxCost.entrySet();
//		Iterator<Map.Entry<String,List<Double>>> it = entrySet.iterator();
//		while(it.hasNext())
//		{
//			Map.Entry<String,List<Double>> me = it.next();
//			bw.write(me.getKey()+", ");
//			for(Double in : me.getValue()) {
//				bw.write(in.doubleValue() + ", ");
//			}
//			bw.write("\r\n");
//		}
//		bw.close();
	}

	//从文件中读取minCost和maxCost
	private static void readMinMaxCost(String fileName) throws IOException {
		String str = null;
		FileInputStream fis = new FileInputStream(OUTPUT_LOCATION + "\\" + fileName);
		InputStreamReader isr = new InputStreamReader(fis);// InputStreamReader 是字节流通向字符流的桥梁,
		BufferedReader br = new BufferedReader(isr);
		try
		{
			while ((str = br.readLine()) != null) {
				String[] s = str.split(", ");
//				System.out.println(s[0]);
				Main.minMaxCost.put(s[0], new ArrayList<Double>());
				Main.minMaxCost.get(s[0]).add(Double.parseDouble(s[1]));
				Main.minMaxCost.get(s[0]).add(Double.parseDouble(s[2]));
			}			
			fis.close();
			isr.close();
			br.close();
		}catch(IOException e){System.out.println(e.getMessage());}		
	}
	
	/**
	 * @param file	xml所在的文件
	 * @param di	第di个deadline间隔
	 * @param fi	重复执行n次，这是第fi次
	 * @param si	task size
	 */
	private static void testForMinMaxCost(String file, int di) {
		// 解析file文件中的工作流
		Workflow wf = new Workflow(file);
	
		Benchmarks benSched = new Benchmarks(wf); // 获得当前工作流的两个Benchmark解，为了计算max min的deadline
		System.out.print("Benchmark-FastSchedule：" + benSched.getFastSchedule());
		System.out.print("Benchmark-CheapSchedule：" + benSched.getCheapSchedule());
		System.out.print("Benchmark-MinCost8Schedule：" + benSched.getMinCost8Schedule());
	
		// 求当前的deadline = min+ (max-min)*deadlineFactor
		double deadlineFactor = 0;
	//	if(di == 0)
	//		deadlineFactor = 1.5;
	//	else
			deadlineFactor = DF_START + DF_INCR * di;
		double deadline = benSched.getFastSchedule().calcMakespan() * deadlineFactor;
		System.out.println("deadlineFactor=" + String.format("%.3f", deadlineFactor) + ", deadline = "
				+ String.format("%.3f", deadline));
		System.out.println();
		
		for (int mi = 0; mi < METHODS.length; mi++) { // method index
			Workflow wf1 = new Workflow(file);
			Scheduler method = null;
			method = METHODS[mi];
			wf1.setDeadline(deadline);
			
			System.out.println("运行算法The current algorithm: " + method.getClass().getCanonicalName());
	
			// 调用算法
			long starTime = System.currentTimeMillis();
			Solution sol = method.schedule(wf1);
	
			long endTime = System.currentTimeMillis();
			double runTime = (double) (endTime - starTime);
	
			String methodName = method.getClass().getName().substring(33);
			if (isPrintExcel)
				em.writeToExcel(ExcelFilePath, sheetName, deadlineFactor, deadline, methodName, mi, sol);
	
			if (sol == null) {
				System.out.println("solution is " + sol + "!\r\n");
				continue;
			}
			int isSatisfied = sol.calcMakespan() <= deadline + E ? 1 : 0;
			List<Integer> result = sol.validateId(wf1);
			if (result.get(0).intValue() == 0) {
				if (isPrintExcel)
					em.writeToExcel(ExcelFilePath, sheetName, result.get(1).intValue(), result.get(0).intValue());
				throw new RuntimeException();
			}
			System.out.println("runtime：" + runTime + "ms;   solution: " + sol);
	
			String fileName = file.substring(file.lastIndexOf("\\")+1, file.indexOf("."));
			if(isSatisfied == 1) {
				double cost = sol.calcCost();
				if(cost < Main.minMaxCost.get(fileName).get(0).doubleValue())
					Main.minMaxCost.get(fileName).set(0, Double.valueOf(cost));
				if(cost > Main.minMaxCost.get(fileName).get(1).doubleValue())
					Main.minMaxCost.get(fileName).set(1, Double.valueOf(cost));
			}
	//		sol.calcCost() / benSched.getCheapSchedule().calcCost();
			
		}
	}

	/**
	 * 计算file工作流(在第si个size下，fileIndex下)在deadline为di松紧下时各算法的结果（成功率，标准cost）
	 * 
	 * @param file
	 *            xml所在的文件
	 * @param di
	 *            第di个deadline间隔
	 * @param fi
	 *            重复执行n次，这是第fi次
	 * @param si
	 *            task size
	 * @param successResult
	 *            在di下个算法的成功率
	 * @param NCResult
	 *            normalized cost
	 * @param refValues
	 * @throws IOException 
	 */
	private static void test(String file, int di, int fi, int si, double[][][] successResult, double[][][] NCResult,
			double[] refValues) throws IOException {
		// 解析file文件中的工作流
		Workflow wf = new Workflow(file);

		Benchmarks benSched = new Benchmarks(wf); // 获得当前工作流的两个Benchmark解，为了计算max min的deadline
		System.out.print("Benchmark-FastSchedule：" + benSched.getFastSchedule());
		System.out.print("Benchmark-CheapSchedule：" + benSched.getCheapSchedule());
		System.out.print("Benchmark-MinCost8Schedule：" + benSched.getMinCost8Schedule());
//		double fastestCost = benSched.getFastSchedule().calcCost();
//		double fastestMakespan = benSched.getFastSchedule().calcMakespan();
//		double slowestCost = benSched.getCheapSchedule().calcCost();
//		double slowestMakespan = benSched.getCheapSchedule().calcMakespan();

		// 求当前的deadline = min+ (max-min)*deadlineFactor
		double deadlineFactor = 0;
//		if(di == 0)
//			deadlineFactor = 1.5;
//		else
			deadlineFactor = DF_START + DF_INCR * di;
		double deadline = benSched.getFastSchedule().calcMakespan() * deadlineFactor;
//		double deadline = benSched.getFastSchedule().calcMakespan()
//				+ (benSched.getCheapSchedule().calcMakespan() - benSched.getFastSchedule().calcMakespan())
//						* deadlineFactor;
		System.out.println("deadlineFactor=" + String.format("%.3f", deadlineFactor) + ", deadline = "
				+ String.format("%.3f", deadline));
		System.out.println();
		for (int mi = 0; mi < METHODS.length; mi++) { // method index
			Workflow wf1 = new Workflow(file);

			Scheduler method = METHODS[mi];
			String methodName = method.getClass().getCanonicalName();
			methodName = methodName.substring(methodName.lastIndexOf(".")+1);
//			if(Main.isCallClassifyMethod && mi == 0) { //我的分类调度算法
			if(methodName.equals("ClassifySchedule")) { //我的分类调度算法
				int a = file.lastIndexOf("\\");
				String key = file.substring(a) + "d" + deadlineFactor;
				String cResult = Main.classifyResult.get(key);
//				if(cResult.equals("1"))
//					method = new Method1uRank();
//				else if(cResult.equals("2"))
//					method = new Method2ProLiS();
//				else if(cResult.equals("3"))
//					method = new Method6();
//				else if(cResult.equals("4"))
//					method = new Method3();
//				else {
//					System.out.println("无效的分类结果");
//					System.exit(0);
//				}
				method = new ClassifySchedule(cResult);
			}
//			else
//				method = METHODS[mi];
			wf1.setDeadline(deadline);
			wf1.setDeadlineFactor(deadlineFactor); // 为HGSA增加
			
			System.out.println("运行算法The current algorithm: " + method.getClass().getCanonicalName());

			// 调用算法
			long starTime = System.currentTimeMillis();
			Solution sol = method.schedule(wf1);

			long endTime = System.currentTimeMillis();
			double runTime = (double) (endTime - starTime);

//			String methodName = method.getClass().getName().substring(33);
			if (isPrintExcel)
				em.writeToExcel(ExcelFilePath, sheetName, deadlineFactor, deadline, methodName, mi, sol);

			if (sol == null) {
				System.out.println("solution is " + sol + "!\r\n");
				continue;
			}
			int isSatisfied = sol.calcMakespan() <= deadline + E ? 1 : 0;
			List<Integer> result = sol.validateId(wf1);
			if (result.get(0).intValue() == 0) {
				if (isPrintExcel)
					em.writeToExcel(ExcelFilePath, sheetName, result.get(1).intValue(), result.get(0).intValue());
				throw new RuntimeException();
			}
			System.out.println("runtime：" + runTime + "ms;   solution: " + sol);

			String fileName = file.substring(file.lastIndexOf("\\")+1, file.indexOf("."));
			successResult[di][mi][fi + si* REPEATED_TIMES] += isSatisfied;
//			NCResult[di][mi][fi + si* REPEATED_TIMES] += sol.calcCost() / benSched.getCheapSchedule().calcCost();
//			NCResult[di][mi][fi + si* REPEATED_TIMES] += Math.abs(sol.calcCost()-slowestCost)/(Math.abs(fastestCost-slowestCost));
			double cost = (sol.calcCost()-Main.minMaxCost.get(fileName).get(0))
					/(Main.minMaxCost.get(fileName).get(1)-Main.minMaxCost.get(fileName).get(0));
			if(onlyCalCostOfFeasible) {
				if(isSatisfied == 1)
					NCResult[di][mi][fi + si*REPEATED_TIMES] += cost;
				else
					NCResult[di][mi][fi + si*REPEATED_TIMES] += 0;
			}
			else
				NCResult[di][mi][fi + si* REPEATED_TIMES] += cost;
			if(isSatisfied == 1) {
				if(cost < 0) { //说明求得了更小的结果
					bwUpdate = new BufferedWriter(new FileWriter(OUTPUT_LOCATION + "\\update_minMaxCost.txt", true));
					bwUpdate.write(fileName+", min"+  Main.minMaxCost.get(fileName).get(0)+ "==> "+ sol.calcCost());
					bwUpdate.close();
				}
				if(sol.calcCost() > Main.minMaxCost.get(fileName).get(1)) {
					bwUpdate = new BufferedWriter(new FileWriter(OUTPUT_LOCATION + "\\update_minMaxCost.txt", true));
					bwUpdate.write(fileName+", max"+  Main.minMaxCost.get(fileName).get(1)+ "==> "+ sol.calcCost());
					bwUpdate.close();
				}
			}
		}
		
		refValues[0] += benSched.getFastSchedule().calcCost();
		refValues[1] += benSched.getFastSchedule().calcMakespan();
		refValues[2] += benSched.getCheapSchedule().calcCost();
		refValues[3] += benSched.getCheapSchedule().calcMakespan();
	}

	private static final java.text.DecimalFormat df = new java.text.DecimalFormat("0.000000");
	static {
		df.setRoundingMode(RoundingMode.HALF_UP);
	}

	private static void printTo(BufferedWriter bw, double[][][] result, String resultName) throws Exception {
		bw.write(resultName + "\r\n");
		for (int di = 0; di <= (DF_END - DF_START) / DF_INCR; di++) {
			String text = df.format(DF_START + DF_INCR * di) + "\t";
			for (int mi = 0; mi < METHODS.length; mi++)
				text += df.format(StatUtils.mean(result[di][mi])) + "\t";
			bw.write(text + "\r\n");
			bw.flush();
		}
		bw.write("\r\n\r\n\r\n");
	}
	private static void printCostOfFeasibleTo(BufferedWriter bw, double[][][] successResult, double[][][] NCResult, String resultName)throws Exception{
		bw.write(resultName + "\r\n");
		for(int di = 0;di<=(DF_END-DF_START)/DF_INCR;di++){
			String text = df.format(DF_START + DF_INCR * di) + "\t";
			for(int mi=0;mi<METHODS.length;mi++) {
				text += df.format(StatUtils.sum(NCResult[di][mi])/StatUtils.sum(successResult[di][mi])) + "\t";
			}
			bw.write(text + "\r\n");
			bw.flush();
		}
		bw.write("\r\n\r\n\r\n");
	}

	private static void printSubDTo(BufferedWriter bw, Workflow w) throws Exception {
		// bw.write("\t");
		// for(Task t : w){
		// bw.write(String.format("%d", t.getId()) + ", ");
		// }
		// bw.write("\r\n\t");
		for (Task t : w) {
			bw.write(String.format("%.3f", t.getSubD()) + ", ");
		}
		bw.write("\r\n");
		bw.flush();
	}

	private static void printRunTime(BufferedWriter bw, Workflow w) throws Exception {

		for (Task t : w) {
			bw.write(String.valueOf(t.getId()));
			bw.write("\t");
			bw.write(String.valueOf(t.getTaskSize() / 4.4) + "\t");
			bw.write("\r\n");
		}
		bw.flush();
	}

	private static void printTT(BufferedWriter bw, Workflow w) throws Exception {
		int size = w.size();
		double[][] tt = new double[size][size];
		for (Task t : w) {
			int pId = t.getId();
			for (Edge e : t.getOutEdges()) {
				Task c = e.getDestination();
				int cId = c.getId();
				tt[pId][cId] = e.getDataSize() * 1.0 / VM.NETWORK_SPEED;
			}
		}

		for (int i = 0; i < size; i++) {
			for (int j = 0; j < size; j++) {
				bw.write(String.valueOf(tt[i][j]) + "\t");
			}
			bw.write("\r\n");
		}
		bw.flush();
	}

	private static void readOptimizedSubD(String filePath) {
		FileInputStream fis = null;
		InputStreamReader isr = null;
		BufferedReader br = null; // 用于包装InputStreamReader,提高处理性能。因为BufferedReader有缓冲的，而InputStreamReader没有。
		try {
			String str = "";
//			String str1 = "";
			String[] s, s1, s2;
			fis = new FileInputStream(filePath);// FileInputStream
			// 从文件系统中的某个文件中获取字节
			isr = new InputStreamReader(fis);// InputStreamReader 是字节流通向字符流的桥梁,
			br = new BufferedReader(isr);// 从字符输入流中读取文件中的内容,封装了一个new InputStreamReader的对象
			while ((str = br.readLine()) != null) {
//				str1 += str + "\n";
				HashMap<Integer, Double> task2SubD = new HashMap<Integer, Double>();
				s = str.split("\t");
				s1 = s[1].split(", ");
				for(int i = 0; i < s1.length; i ++) {
					s2 = s1[i].split(": ");
					task2SubD.put(Integer.valueOf(s2[0]), Double.valueOf(s2[1]));
				}
				type2subD.put(s[0].trim(), task2SubD);
			}
//			System.out.println(str1);// 打印出str1
		} catch (FileNotFoundException e) {
			System.out.println("找不到指定文件");
		} catch (IOException e) {
			System.out.println("读取文件失败");
		} finally {
			try {
				br.close();
				isr.close();
				fis.close();
				// 关闭的时候最好按照先后顺序关闭最后开的先关闭所以先关s,再关n,最后关m
			} catch (IOException e) {
				e.printStackTrace();
			}
		}

	}
	
	/**
	 * 从生成的工作流中随机选择使用的工作流
	 * @param eachSizeNum 工作流size有30、50、100、1000四种，每种的个数
	 * @param eachSizeSelNum 工作流size有30、50、100、1000四种，每种被选择的个数
	 * @throws IOException 
	 */
	private static void selectWorkflows(int eachSizeNum, int eachSizeSelNum) throws IOException {
//		String[] types = { "CyberShake","Epigenomics","Inspiral", "Montage", "Sipht"};	
//		int[] sizes = { 30, 50, 100, 1000};
		
		for (int si = 0; si < SIZES.length; si++) { // size index
			List<String> selectedWorkflows = new ArrayList<String>();
			int size = SIZES[si];
			for(int num = 0; num < eachSizeSelNum; num++) {
				int typeIndex = (int)(Math.random()*Main.WORKFLOWS.length); //随机选择某个类型的工作流，如CyberShake
				int fileIndex = (int)(Math.random()*eachSizeNum);
				String file = Main.WORKFLOWS[typeIndex] +"_"+ size + "." + fileIndex + ".xml";
				while(selectedWorkflows.contains(file)) {
					fileIndex = (int)(Math.random()*100);
					file = Main.WORKFLOWS[typeIndex] +"_"+ size + "." + fileIndex + ".xml";
				}
				selectedWorkflows.add(file);
			}
			BufferedWriter bw = new BufferedWriter(new FileWriter(OUTPUT_LOCATION + "\\" + "selectedWorkflows_" + size + ".txt"));
			for(String de : selectedWorkflows)
			{
				bw.write(de);
				bw.write("\r\n");
			}
			bw.close();
		}
	}
	
	/**获取工作流测试集
	 * @throws IOException 
	 * @throws ClassNotFoundException */
	public static List<String> getWorkflowListFromFile(String filename) throws IOException, ClassNotFoundException
	{
		List<String> w_List = new ArrayList<String>();
		String w = null;
		FileInputStream fis = new FileInputStream(filename);
		InputStreamReader isr = new InputStreamReader(fis);// InputStreamReader 是字节流通向字符流的桥梁,
		BufferedReader br = new BufferedReader(isr);
		try
		{
			for(int i=0; i< WorkflowNum; i++)
			{
				w = br.readLine();
				w_List.add(w);
			}			
			fis.close();
			isr.close();
			br.close();
		}catch(IOException e){System.out.println(e.getMessage());}		
		return w_List;
	}
	
	/**获取工作流的分类结果
	 * @throws IOException 
	 * @throws ClassNotFoundException */
	public static List<String> getClassifyResultFromFile(String filename) throws IOException, ClassNotFoundException
	{
		List<String> w_List = new ArrayList<String>();
		String str = null;
		FileInputStream fis = new FileInputStream(filename);
		InputStreamReader isr = new InputStreamReader(fis);// InputStreamReader 是字节流通向字符流的桥梁,
		BufferedReader br = new BufferedReader(isr);
		try
		{
			while ((str = br.readLine()) != null) {
				String[] s = str.split(", ");
				Main.classifyResult.put(s[0] + s[1], s[2]);
			}			
			fis.close();
			isr.close();
			br.close();
		}catch(IOException e){System.out.println(e.getMessage());}		
		return w_List;
	}
	
//	/**
//	 * 在Readfile读subD, 分别写入writeFile1和writeFile2
//	 * @param Readfile
//	 * @param writeFile1
//	 * @param  
//	 */
//	private static void readWriteOptimizedSubD(String Readfile, String writeFile1, String writeFile2) {
//		FileInputStream fis = null;
//		InputStreamReader isr = null;
//		BufferedReader br = null; // 用于包装InputStreamReader,提高处理性能。因为BufferedReader有缓冲的，而InputStreamReader没有。
//		try {
//			String str = "";
////			String str1 = "";
//			String[] s, s1, s2;
//			fis = new FileInputStream(Readfile);// FileInputStream
//			// 从文件系统中的某个文件中获取字节
//			isr = new InputStreamReader(fis);// InputStreamReader 是字节流通向字符流的桥梁,
//			br = new BufferedReader(isr);// 从字符输入流中读取文件中的内容,封装了一个new InputStreamReader的对象
//			while ((str = br.readLine()) != null) {
////				str1 += str + "\n";
//				HashMap<Integer, Double> task2SubD = new HashMap<Integer, Double>();
//				s = str.split("\t");
//				s1 = s[1].split(", ");
//				for(int i = 0; i < s1.length; i ++) {
//					s2 = s1[i].split(": ");
//					task2SubD.put(Integer.valueOf(s2[0]), Double.valueOf(s2[1]));
//				}
//				type2subD.put(s[0].trim(), task2SubD);
//			}
////			System.out.println(str1);// 打印出str1
//		} catch (FileNotFoundException e) {
//			System.out.println("找不到指定文件");
//		} catch (IOException e) {
//			System.out.println("读取文件失败");
//		} finally {
//			try {
//				br.close();
//				isr.close();
//				fis.close();
//				// 关闭的时候最好按照先后顺序关闭最后开的先关闭所以先关s,再关n,最后关m
//			} catch (IOException e) {
//				e.printStackTrace();
//			}
//		}
//
//	}
	
}