package cloud.workflowScheduling.idea.subDMethods;

import java.io.IOException;
import java.util.*;

import cloud.workflowScheduling.*;
import cloud.workflowScheduling.idea.classificationScheduling.EvaluateYLW;
import cloud.workflowScheduling.methods.Scheduler;
import cloud.workflowScheduling.setting.*;


public class Method3 implements Scheduler {
	
	private int workflowDepth; //即入口任务的depth
	private double alphaDeadline; 
	private double deadlineFactor;
	private Map<Integer, List<Task>> taskLevelSet;
	private List<Double> initialSubDlevel;
	private List<Double> fbsSubDlevel;
	public Method3(){

	}

	public Solution schedule(Workflow wf) {
		//调度前，先初始化成员变量
		workflowDepth = 0; //即入口任务的depth
		alphaDeadline = 0.0; 
		deadlineFactor = 0.0;
		taskLevelSet = new LinkedHashMap<Integer, List<Task>>();
		initialSubDlevel = new ArrayList<Double>();
		fbsSubDlevel = new ArrayList<Double>();
		
		wf.calcTaskLevels();	
		
		List<Task> tasks = new ArrayList<Task>(wf);
		Collections.sort(tasks, new Task.BLevelComparator()); 	
		Collections.reverse(tasks);	//sort based on URank, larger first
		
		Task entryTask = wf.get(0);
		
		//计算任务depth, 从出口任务开始
		List<Task> exits = new ArrayList<Task>();
		for(Task t : wf) {
			if(t.getOutEdges().isEmpty())
				exits.add(t);
		}
		for (Iterator it = exits.iterator(); it.hasNext();) {
            Task task = (Task) it.next();
            task.setDepthFromExit(task, 1);
        }	
		this.workflowDepth = entryTask.getDepth();
		
		//计算TLS
		for(int i = this.workflowDepth; i > 0; i--){	
			taskLevelSet.put(Integer.valueOf(i), new ArrayList<Task>());
		}
		for(int i = 0; i < wf.size(); i++){
			Task task = wf.get(i);
			int depth = task.getDepth();
			taskLevelSet.get(Integer.valueOf(depth)).add(task);
		}
		
		//初始化每层level的subD并计算EFT（文献中的计算EFT考虑了level的subD），公式13、14
		for(int i = 0; i < this.workflowDepth+1; i++){	
			initialSubDlevel.add(-1.0);
			this.fbsSubDlevel.add(-1.0);
		}
		double bestVMSpeed = VM.SPEEDS[VM.FASTEST];
		entryTask.setEST(0);
		entryTask.setEFT(0);
		initialSubDlevel.set(entryTask.getDepth(), 0.0); 
		for(Map.Entry<Integer, List<Task>> entry : taskLevelSet.entrySet()){ //按照level遍历所有任务
			if(entry.getKey().equals(this.workflowDepth))
				continue;
			else { //遍历当前level中所有任务，计算其EFT, 并计算当前level的subD
				double subD = 0.0;
				List<Task> taskSet = entry.getValue();
				for(Task task : taskSet) {
					double EST = this.initialSubDlevel.get(task.getDepth()+1); //初始化任务所在level的上一级level的subD
					for(Edge e: task.getInEdges()){
						Task parent = e.getSource();
						double startTime = e.getDataSize()/VM.NETWORK_SPEED;
						startTime += parent.getEFT();
						EST = Math.max(EST, startTime);				//determine EST
					}
					task.setEST(EST);
					task.setEFT(EST + task.getTaskSize() / bestVMSpeed);
					
					subD = Math.max(subD, task.getEFT());
				}
				this.initialSubDlevel.set(entry.getKey().intValue(), subD);
			}
		}
		
		//计算公式15
		this.fbsSubDlevel.set(this.workflowDepth, 0.0);
		this.alphaDeadline = this.fbsSubDlevel.get(workflowDepth)*this.taskLevelSet.get(Integer.valueOf(workflowDepth)).size();
		for(int i = this.workflowDepth-1; i > 0; i--){
			this.fbsSubDlevel.set(i, this.initialSubDlevel.get(i)-this.initialSubDlevel.get(i+1));
			this.alphaDeadline += this.fbsSubDlevel.get(i)*this.taskLevelSet.get(Integer.valueOf(i)).size();
		}
		
		//公式16
		this.deadlineFactor = (wf.getDeadline()-this.initialSubDlevel.get(1))/this.alphaDeadline;
		
		//公式17 计算每一层的subD 
		//论文公式，无累加，是错的，下面采用的是累加的方法进行纠正，或许也可以通过更新EFT来纠正
//		for(Map.Entry<Integer, List<Task>> entry : taskLevelSet.entrySet()){ //按照level遍历所有任务
//			double subDlevel = 0; //根据工作流deadline计算的每层的实际subD
//			int level = entry.getKey().intValue();
//			List<Task> taskSet = entry.getValue();
//			subDlevel = this.initialSubDlevel.get(level) + 
//					this.deadlineFactor*this.fbsSubDlevel.get(level)*this.taskLevelSet.get(Integer.valueOf(level)).size();
//			for(Task task : taskSet) {
//				task.setSubD(subDlevel);
//			}	
//		}
		//论文公式，有累加
		double sumFbsSubDlevel = 0; //每一层level长度的累加
		for(Map.Entry<Integer, List<Task>> entry : taskLevelSet.entrySet()){ //按照level遍历所有任务
			double subDlevel = 0; //根据工作流deadline计算的每层的实际subD
			int level = entry.getKey().intValue();
			List<Task> taskSet = entry.getValue();
			sumFbsSubDlevel += this.fbsSubDlevel.get(level)*this.taskLevelSet.get(Integer.valueOf(level)).size();
			subDlevel = this.initialSubDlevel.get(level) + this.deadlineFactor*sumFbsSubDlevel;
			for(Task task : taskSet) {
				task.setSubD(subDlevel);
			}	
		}		
		
		return buildViaTaskList(wf, tasks, wf.getDeadline());
		
	}
	
	//build a solution based on a task ordering.
	//that is, for a given task ordering, distribute deadline and select services here
	public Solution buildViaTaskList(Workflow wf, List<Task> tasks, double deadline) {
//		try {
//			EvaluateYLWSample.subD[0].write(String.format("%15s", "Method3: "));
//		} catch (IOException e1) {
//			// TODO Auto-generated catch block
//			e1.printStackTrace();
//		}
		
		int violationCount = 0;		// test code
		Solution solution = new Solution();
		
		for(int i = 1; i < tasks.size(); i++){		
			Task task = tasks.get(i);
			int level = task.getDepth();
			double proSubDeadline = task.getSubD();
			
			task.setSubD(proSubDeadline);
//			try {
//				EvaluateYLWSample.subD[0].write(task.getId() + ": " + String.format("%.3f", proSubDeadline) + ", ");
//			} catch (IOException e) {
//				// TODO Auto-generated catch block
//				e.printStackTrace();
//			}
			Allocation alloc = getMinCostVM(task, solution,proSubDeadline, i);

			//当CPLength>deadline时，子期限的划分可能导致EFT>subDeadline；所以必须考虑子期限不满足的情况：此时选择minimal EFT的VM
			if(alloc == null){			//select a vm which allows EFT
				alloc = getMinEFTVM(task, solution, proSubDeadline, i);
				
//				VM vm = alloc.getVM();
//				while(alloc.getFinishTime() > proSubDeadline + Evaluate.E && vm.getType() < VM.FASTEST){
//					solution.updateVM(vm);			//upgrade若进行整个解的更新；复杂度将增长太多。
//					alloc.setStartTime(solution.calcEST(task, vm));
//					alloc.setFinishTime(solution.calcEST(task, vm) + task.getTaskSize()/vm.getSpeed());
//				}
				if(alloc.getFinishTime() > proSubDeadline + EvaluateYLW.E)
					violationCount ++;
			}
			if(i == 1)		//after allocating task_1, allocate entryTask to the same VM 
				solution.addTaskToVM(alloc.getVM(), tasks.get(0), alloc.getStartTime(), true);
			solution.addTaskToVM(alloc.getVM(), task, alloc.getStartTime(), true);	//allocate

		}
//		try {
//			EvaluateYLWSample.subD[0].write("\r\n");
//		} catch (IOException e) {
//			// TODO Auto-generated catch block
//			e.printStackTrace();
//		}
//		if(violationCount > 0)
//			System.out.println("Number of sub-deadline violation: " + violationCount);
		
		return solution;
	}
	
	// select a vm that meets sub-deadline and minimizes the cost
	//candidate services include all the services that have been used (i.e., R), 
	//			and those that have not been used but can be added any time (one service for each type)
	private Allocation getMinCostVM(Task task, Solution solution, double subDeadline, int taskIndex){
		double minIncreasedCost = Double.MAX_VALUE;	//increased cost for one VM is used here, instead of total cost
		VM selectedVM = null;
		double selectedStartTime = 0;
		
		double maxOutTime = 0;	//maxTransferOutTime
		for(Edge e : task.getOutEdges())
			maxOutTime = Math.max(maxOutTime, e.getDataSize());
		maxOutTime /= VM.NETWORK_SPEED;
		
		double startTime, finishTime;
		// traverse VMs in solution to find a vm that meets sub-deadline and minimizes the cost
		for(VM vm : solution.keySet()){	
			startTime = solution.calcEST(task, vm); 
			finishTime = startTime + task.getTaskSize()/vm.getSpeed();
			if(finishTime > subDeadline + EvaluateYLW.E)   //sub-deadline not met
				continue;
			
			double newVMPeriod = finishTime + maxOutTime - solution.getVMLeaseStartTime(vm);
			double newVMTotalCost = Math.ceil(newVMPeriod/VM.INTERVAL) * vm.getUnitCost();
			double increasedCost = newVMTotalCost - solution.calcVMCost(vm);  // oldVMTotalCost
			if(increasedCost < minIncreasedCost){ 
				minIncreasedCost = increasedCost;
				selectedVM = vm;
				selectedStartTime = startTime;
			}
		}

		//test whether a new VM can meet the sub-deadline and (or) reduce increasedCost; if so, add this new VM
		int selectedI = -1;				
		startTime = taskIndex==1 ? VM.LAUNCH_TIME : solution.calcEST(task, null);
		for(int k = 0 ; k<VM.TYPE_NO; k++){
			finishTime = startTime + task.getTaskSize()/VM.SPEEDS[k];
			if(finishTime > subDeadline + EvaluateYLW.E)	//sub-deadline not met
				continue;
			
			double increasedCost = Math.ceil((finishTime - startTime)/VM.INTERVAL) * VM.UNIT_COSTS[k];
			if(increasedCost < minIncreasedCost){
				minIncreasedCost = increasedCost;
				selectedI = k;
				selectedStartTime = startTime;
			}
		}
		if(selectedI != -1)
			selectedVM = new VM(selectedI);
		
		if(selectedVM == null)
			return null;
		else
			return new Allocation(selectedVM, task, selectedStartTime);
	}
	
	//select a VM from R which minimizes the finish time of the task
	//here, candidates only include services from R if R is not null
	private Allocation getMinEFTVM(Task task, Solution solution, double subDeadline, int taskIndex){
		VM selectedVM = null;				
		double selectedStartTime = 0;
		double minEFT = Double.MAX_VALUE;
		
		double startTime, finishTime;
		// traverse VMs in solution to find a vm that minimizes EFT
		for(VM vm : solution.keySet()){			
			startTime = solution.calcEST(task, vm); 
			finishTime = startTime + task.getTaskSize()/vm.getSpeed();
			if(finishTime < minEFT){
				minEFT = finishTime;
				selectedVM = vm;
				selectedStartTime = startTime;
			}
		}

		// if solution has no VMs 
		if(selectedVM==null ){		// logically, it is equal to "solution.keySet().size()==0"
			startTime = taskIndex==1 ? VM.LAUNCH_TIME : solution.calcEST(task, null);
			finishTime = startTime + task.getTaskSize()/VM.SPEEDS[VM.FASTEST];
			if(finishTime < minEFT){
				minEFT = finishTime;
				selectedStartTime = startTime;
				selectedVM = new VM(VM.FASTEST);
			}
		}
		return  new Allocation(selectedVM, task, selectedStartTime);
	}
}