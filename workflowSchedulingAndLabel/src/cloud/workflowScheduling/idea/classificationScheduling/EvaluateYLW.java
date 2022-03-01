package cloud.workflowScheduling.idea.classificationScheduling;

import java.io.*;
import java.math.*;
import java.util.*;

import org.apache.commons.math3.stat.*;

import cloud.workflowScheduling.idea.subDMethods.*;
import cloud.workflowScheduling.methods.*;
import cloud.workflowScheduling.setting.*;

//为每个文件打label
//每个算法重复执行10次，因为ProLiS结果变化 EvaluateYLW2
public class EvaluateYLW {
	// because in Java float numbers can not be precisely stored, a very small
	// number E is added before testing whether deadline is met
	public static final double E = 0.0000001;

	private static double DF_START = 1, DF_INCR = 1, DF_END = 10;
	
	private static final int REPEATED_TIMES = 10;
	private static final int FILE_INDEX_MAX = 400; //100或200， 每类生成了100个xml或200个xml
	private static final int[] SIZES = { 30, 50, 100, 1000};

	//算法new之后一直在用，成员变量的值会保存上一次的执行，所以进入算法后，先初始化成员变量
	private static final Scheduler[] METHODS = {new Method1uRank(), new Method2ProLiS(), new Method6(), new Method3(), }; 
	private static final String[] WORKFLOWS = { "CyberShake","Epigenomics","Inspiral", "Montage", "Sipht"};

//	static final String WORKFLOW_LOCATION = ".\\SyntheticWorkflows";
	static final String WORKFLOW_LOCATION = "E:\\0Work\\1Ideas\\workflowClassificationAndScheduling\\workflowScheduling\\workflowGenerator\\WorkflowGenerator\\bharathi\\generatedWorkflows";
	static final String OUTPUT_LOCATION = ".\\result\\repeat";
	static final String ExcelFilePath = ".\\result\\repeat\\solution.xls";
//	static final String SubDPath = ".\\result\\PSO优化subD对50-100-1000\\subDeadline1-opSubD-PSO-proLiSURank-50-100-100.txt";
	public static ExcelManage em;
	public static String sheetName;
	public static boolean isPrintExcel = false; // true false

//	public static BufferedWriter[] subD = new BufferedWriter[20];
	public static HashMap<String, HashMap<Integer, Double>> type2subD = new HashMap<String, HashMap<Integer, Double>>();
	public static Map<String, Integer> workflow2lable = new LinkedHashMap<String, Integer>();
	public static List<Double> deadlines = new ArrayList<Double>();
	public static List<Double> deadlineFactors = new ArrayList<Double>();
	public static Map<String, String> workflow2staticalLabel = new LinkedHashMap<String, String>(); //工作流对应的统计最好方法
	public static Map<String, Double> costOnStaticalLabel = new LinkedHashMap<String, Double>(); //每个文件在统计最好方法下的cost
	
	//为每个文件打label
	public static void main(String[] args) throws Exception {
//		readOptimizedSubD(SubDPath);

//		String fileName1 = OUTPUT_LOCATION + "\\workflow2staticalLabel.txt"; 
//		String fileName2 = OUTPUT_LOCATION + "\\costOnStaticalLabel.txt";
//		read(fileName1, fileName2);

		if (isPrintExcel)
			ExcelManage.clearExecl(ExcelFilePath);
		int deadlineNum = (int) ((DF_END - DF_START) / DF_INCR + 1);
		for (String workflow : WORKFLOWS) {
			for (int si = 0; si < SIZES.length; si++) { // size index
				int size = SIZES[si];
				sheetName = workflow + "_" + size;
				if (isPrintExcel)
					em = ExcelManage.initExecl(ExcelFilePath, sheetName);
//				for (int di = 0; di <= (DF_END - DF_START) / DF_INCR; di++) { // deadline index
				for(int fi = 0;fi<FILE_INDEX_MAX;fi++){	 // workflow file index
					for (int di = 0; di <= (DF_END - DF_START) / DF_INCR; di++) { // deadline index
//						String file = WORKFLOW_LOCATION + "\\" + workflow + "\\" + workflow + ".n." + size + "." + fi + ".dax";
						String file = WORKFLOW_LOCATION + "\\" + workflow +"_"+ size + "." + fi + ".xml";
						test(file, di, fi, si);
					}
				}
			}

		}
		BufferedWriter bw1 = new BufferedWriter(new FileWriter(OUTPUT_LOCATION + "\\" + "workflow2label.txt"));
		Set<Map.Entry<String,Integer>> entrySet = workflow2lable.entrySet();
		Iterator<Map.Entry<String,Integer>> it = entrySet.iterator();
		while(it.hasNext())
		{
			Map.Entry<String,Integer> me = it.next();
			bw1.write(me.getKey()+", "+me.getValue().intValue());
			bw1.write("\r\n");
		}
		bw1.close();
		
		BufferedWriter bw2 = new BufferedWriter(new FileWriter(OUTPUT_LOCATION + "\\" + "deadlines.txt"));
		for(Double de : deadlines)
		{
			bw2.write(de.toString());
			bw2.write("\r\n");
		}
		bw2.close();
		
		BufferedWriter bw3 = new BufferedWriter(new FileWriter(OUTPUT_LOCATION + "\\" + "deadlineFactors.txt"));
		for(Double de : deadlineFactors)
		{
			bw3.write(de.toString());
			bw3.write("\r\n");
		}
		bw3.close();
		
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
	 */
	private static void test(String file, int di, int fi, int si) {
		// 解析file文件中的工作流
		Workflow wf = new Workflow(file);

		Benchmarks benSched = new Benchmarks(wf); // 获得当前工作流的两个Benchmark解，为了计算max min的deadline
		System.out.print("Benchmark-FastSchedule：" + benSched.getFastSchedule());
		System.out.print("Benchmark-CheapSchedule：" + benSched.getCheapSchedule());
		System.out.print("Benchmark-MinCost8Schedule：" + benSched.getMinCost8Schedule());

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
		EvaluateYLW.deadlines.add(Double.valueOf(deadline));
		EvaluateYLW.deadlineFactors.add(Double.valueOf(deadlineFactor));
		String fileName = file.substring(file.lastIndexOf("\\")) + ", d" + deadlineFactor;

		System.out.println();
		double bestCost = Double.MAX_VALUE;
		int bestMethod = Integer.MAX_VALUE;
		int bestIsDeadline = 0;
		for (int mi = 0; mi < METHODS.length; mi++) { // method index
			Workflow wf1 = new Workflow(file);

			Scheduler method = METHODS[mi];
			wf1.setDeadline(deadline);
			wf1.setDeadlineFactor(deadlineFactor); // 为HGSA增加
			
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
			
			if(isSatisfied == 1 && mi != 1) { //mi=1是包含ProLiS的方法，该方法特殊处理
				double currentCost = sol.calcCost();
				if(currentCost < bestCost) {
					bestCost = currentCost;
					bestMethod = mi + 1;
				}
			}
			
			if(mi == 1) { //mi=1是包含ProLiS的方法，该方法再重新跑9次 
				double bestCostProLiS = sol.calcCost(); //设置为当前花费
				int satisfyRate = sol.calcMakespan() <= deadline + E ? 1 : 0;
				for (int timeI = 1; timeI < REPEATED_TIMES; timeI++) { //如果是ProLiS重复运行10次
					wf1 = new Workflow(file);

					method = METHODS[mi];
					wf1.setDeadline(deadline);
					wf1.setDeadlineFactor(deadlineFactor); // 为HGSA增加
					
					System.out.println("运行算法The current algorithm: " + method.getClass().getCanonicalName());

					// 调用算法
					starTime = System.currentTimeMillis();
					sol = method.schedule(wf1);

					endTime = System.currentTimeMillis();
					runTime = (double) (endTime - starTime);

					methodName = method.getClass().getName().substring(33);
					if (isPrintExcel)
						em.writeToExcel(ExcelFilePath, sheetName, deadlineFactor, deadline, methodName, mi, sol);

					if (sol == null) {
						System.out.println("solution is " + sol + "!\r\n");
						continue;
					}
					isSatisfied = sol.calcMakespan() <= deadline + E ? 1 : 0;
					result = sol.validateId(wf1);
					if (result.get(0).intValue() == 0) {
						if (isPrintExcel)
							em.writeToExcel(ExcelFilePath, sheetName, result.get(1).intValue(), result.get(0).intValue());
						throw new RuntimeException();
					}
					System.out.println("runtime：" + runTime + "ms;   solution: " + sol);
					
					if(isSatisfied == 1) { //mi=1是包含ProLiS的方法，该方法特殊处理
						satisfyRate++;
						double currentCost = sol.calcCost(); 
						bestCostProLiS += currentCost;
					}
				}
				if(satisfyRate == 10) { //ProLiS重复运行10次，都能获得可行解
					bestCostProLiS = bestCostProLiS/10;
					if(bestCostProLiS < bestCost) {
						bestCost = bestCostProLiS;
						bestMethod = mi + 1;
					}
				}
			}
		}
//		String key = file.substring(file.lastIndexOf("\\")+1, file.indexOf(".")) +", d"+deadlineFactor;
//		int staticalBestMethod = Integer.valueOf(EvaluateYLW2.workflow2staticalLabel.get(key));
		if(bestMethod == Integer.MAX_VALUE) //如果没有满足约束的方法，随机选择一个方法
			bestMethod = 1; //(int)(Math.random()*4)+1;
//			bestMethod = staticalBestMethod;
		
//		if(staticalBestMethod != bestMethod) {
//			double performanceDegradationRate = (EvaluateYLW2.costOnStaticalLabel.get(fileName).doubleValue() - bestCost)/bestCost;
//			if(performanceDegradationRate < 0.1) //相对于最好的方法，性能变差率小于10%，还是取统计最好结果
//				bestMethod = staticalBestMethod;
//		}
		workflow2lable.put(fileName, bestMethod);
//		workflow2lable.put(fileName, staticalBestMethod);
		
	}

	private static final java.text.DecimalFormat df = new java.text.DecimalFormat("0.000");
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
	
	private static void read(String fileName1, String fileName2) throws FileNotFoundException {

		String str = null;
		FileInputStream fis = new FileInputStream(fileName1);
		InputStreamReader isr = new InputStreamReader(fis);
		BufferedReader br = new BufferedReader(isr);
		try
		{
			while ((str = br.readLine()) != null) {
				String[] s = str.split(", ");
				EvaluateYLW.workflow2staticalLabel.put(s[0] + ", " +s[1], s[2]);
			}			
			fis.close();
			isr.close();
			br.close();
		}catch(IOException e){System.out.println(e.getMessage());}		
		
		fis = new FileInputStream(fileName2);
		isr = new InputStreamReader(fis);
		br = new BufferedReader(isr);
		try
		{
			while ((str = br.readLine()) != null) {
				String[] s = str.split(", ");
				EvaluateYLW.costOnStaticalLabel.put(s[0] + ", " +s[1], Double.valueOf(s[2]));
			}			
			fis.close();
			isr.close();
			br.close();
		}catch(IOException e){System.out.println(e.getMessage());}	
		
	}
	
}