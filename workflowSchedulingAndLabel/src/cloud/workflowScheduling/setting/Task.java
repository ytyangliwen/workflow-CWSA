package cloud.workflowScheduling.setting;

import java.io.Serializable;
import java.util.*;

public class Task implements Serializable{

	private static int internalId = 0;
	static void resetInternalId(){		//invoked by the constructor of Workflow
		internalId = 0;
	}
	private int id;
	private String name;
	private double taskSize;
	public double averageET;

	//adjacent list to store workflow graph
	//由于子边的终端之间可能也存在父子关系，所以这些edge都是按照其终端对应的拓扑顺序进行排序的;  通过workflow中的refine函数实现
	private List<Edge> outEdges = new ArrayList<Edge>();	
	private List<Edge> inEdges = new ArrayList<Edge>();
	private int topoCount = 0;			//used for topological sort
	
	private double bLevel; 	//blevel
	private double tLevel;	//tLevel
	private double sLevel;
	private double ALAP;
	private double pURank; //Probabilistic Upward  Rank 
	
	private double subD = 0;
	private double dRank = 0;
	private double fastET = 0; //最快执行时间
	private int depth =0;
	private double inData = 0;

	public Task(String name, double taskSize) {
		this.id = internalId++;
		this.name = name;
		this.taskSize = taskSize;
	}

	//-------------------------------------getters&setters--------------------------------
	public int getId() {
		return id;
	}
	public String getName() {
		return name;
	}
	public double getTaskSize() {
		return taskSize;
	}
	public double getbLevel() {
		return bLevel;
	}
	public void setbLevel(double bLevel) {
		this.bLevel = bLevel;
	}
	public double gettLevel() {
		return tLevel;
	}
	public void settLevel(double tLevel) {
		this.tLevel = tLevel;
	}
	public double getsLevel() {
		return sLevel;
	}
	public void setsLevel(double sLevel) {
		this.sLevel = sLevel;
	}
	public double getALAP() {
		return ALAP;
	}
	public void setALAP(double aLAP) {
		ALAP = aLAP;
	}
	public List<Edge> getOutEdges() {
		return outEdges;
	}
	public List<Edge> getInEdges() {
		return inEdges;
	}
	public void insertInEdge(Edge e){
		if(e.getDestination()!=this)
			throw new RuntimeException();	
		inEdges.add(e);
	}
	public void insertOutEdge(Edge e){
		if(e.getSource()!=this)
			throw new RuntimeException();
		outEdges.add(e);
	}
	public int getTopoCount() {
		return topoCount;
	}
	public void setTopoCount(int topoCount) {
		this.topoCount = topoCount;
	}
	public double getpURank() {
		return pURank;
	}
	public void setpURank(double pURank) {
		this.pURank = pURank;
	}
	public double getSubD() {
		return subD;
	}
	public void setSubD(double subD) {
		this.subD = subD;
	}
	public double getDRank() {
		return this.dRank;
	}
	public void setDRank(double dRank) {
		this.dRank = dRank;
	}
	public void calDRank(Solution s) {
		HashMap<Task, Allocation> remap = s.getRevMapping();
		double tt = 0; //传输时间
		Allocation cAlloc = remap.get(this);
		double arrivalTime = 0;
		for(Edge inEdge : this.getInEdges()){
			Task parent = inEdge.getSource();
			Allocation pAlloc = remap.get(parent);
			if(pAlloc.getVM() == cAlloc.getVM())
				tt = 0;
			else
				tt = inEdge.getDataSize() / VM.NETWORK_SPEED;
//			arrivalTime = Math.max(arrivalTime, 
//					parent.getDRank() + parent.getTaskSize() / pAlloc.getVM().getSpeed() + tt);
			arrivalTime = Math.max(arrivalTime, 
					parent.getDRank() + parent.getFastET() + tt);
		}
		this.setDRank(arrivalTime);	
	}
	public void updateURank(Solution s) {
		HashMap<Task, Allocation> remap = s.getRevMapping();
		VM vm = remap.get(this).getVM();
		this.setbLevel(this.getbLevel() + this.taskSize/vm.getSpeed() - this.getFastET());
	}
	public void updateParentsURank(Solution s) {
		if(this.name.equals("entry"))
			return;
		
		HashMap<Task, Allocation> remap = s.getRevMapping();
		double tt = 0; //传输时间
		
		for(Edge inEdge : this.getInEdges()){
			Task parent = inEdge.getSource();	
			Allocation pAlloc = remap.get(parent);
			double bLevel = 0;	
			for(Edge outEdge : parent.getOutEdges()){
				Task child = outEdge.getDestination();
				if(remap.containsKey(child)) {
					Allocation cAlloc = remap.get(child);
					if(pAlloc.getVM() == cAlloc.getVM())
						tt = 0;
					else
						tt = outEdge.getDataSize() / VM.NETWORK_SPEED;
				}
				else 
					tt = outEdge.getDataSize() / VM.NETWORK_SPEED;
				bLevel = Math.max(bLevel, child.getbLevel() + tt);
			}
			parent.setbLevel(bLevel + parent.getTaskSize() / pAlloc.getVM().getSpeed());
//			parent.setbLevel(bLevel + parent.getFastET());
		}
		
		for(Edge inEdge : this.getInEdges()){
			Task parent = inEdge.getSource();
			parent.updateParentsURank(s);
		}
	}
	public double getFastET() {
		return this.fastET;
	}
	public void setFastET(double fastet) {
		this.fastET = fastet;
	}
	
	public int getDepth() {
		return this.depth;
	}
	public void setDepth(int depth) {
		this.depth = depth;
	}
	
	public void setDepth(Task task, int depth) {
        if (depth > task.getDepth()) {
            task.setDepth(depth);
        }
        for(Edge e : task.getOutEdges()){	// for each node m with an edge e from n to m do
			Task cTask = e.getDestination();
			setDepth(cTask, task.getDepth() + 1);
        }
    }
	public void setDepthFromExit(Task task, int depth) {
        if (depth > task.getDepth()) {
            task.setDepth(depth);
        }
        for(Edge e : task.getInEdges()){	// for each node m with an edge e from n to m do
			Task pTask = e.getSource();
			setDepthFromExit(pTask, task.getDepth() + 1);
        }
    }
	
	public double getIndata() {
		return this.inData;
	}
	public void setIndata(double inD) {
		this.inData = inD;
	}
	public void calIndata() {
		double tt = 0; //传输时间
		double maxTT = 0;
		for(Edge inEdge : this.getInEdges()){
			Task parent = inEdge.getSource();
			tt = inEdge.getDataSize() / VM.NETWORK_SPEED;
			maxTT = Math.max(maxTT,  tt);
		}
		this.setIndata(maxTT);	
	}

	
	
	//-------------------------------------overrides--------------------------------
	public String toString() {
//		return "Task [id=" + name + ", taskSize=" + taskSize +"]";
		return "Task [id=" + id + ", taskSize=" + taskSize +"]";
//		return id;
	}

	//-------------------------------------comparators--------------------------------
	public static class BLevelComparator implements Comparator<Task>{
		public int compare(Task o1, Task o2) {
			// to keep entry node ranking last, and exit node first
			if(o1.getName().equals("entry") || o2.getName().equals("exit"))	
				return 1;
			if(o1.getName().equals("exit") || o2.getName().equals("entry"))	
				return -1;
			if(o1.getbLevel()>o2.getbLevel())
				return 1;
			else if(o1.getbLevel()<o2.getbLevel())
				return -1;
			else{
				return 0;
			}
		}
	}
	public static class PURankComparator implements Comparator<Task>{	
		public int compare(Task o1, Task o2) {
			// to keep entry node ranking last, and exit node first
			if(o1.getName().equals("entry") || o2.getName().equals("exit"))	
				return 1;
			if(o1.getName().equals("exit") || o2.getName().equals("entry"))	
				return -1;
			if(o1.getpURank()>o2.getpURank())
				return 1;
			else if(o1.getpURank()<o2.getpURank())
				return -1;
			else{
				return 0;
			}
		}
	}
	public static class TLevelComparator implements Comparator<Task>{
		public int compare(Task o1, Task o2) {
			if(o1.getName().equals("entry") || o2.getName().equals("exit"))	
				return -1;
			if(o1.getName().equals("exit") || o2.getName().equals("entry"))	
				return 1;
			if(o1.gettLevel()>o2.gettLevel())
				return 1;
			else if(o1.gettLevel()<o2.gettLevel())
				return -1;
			else{
				return 0;
			}
		}
	}
	// used to calculate the largest number of parallel tasks in workflow
	public static class ParallelComparator implements Comparator<Task>{
		public int compare(Task o1, Task o2) {
			int d1 = o1.getOutEdges().size() - o1.getInEdges().size();
			int d2 = o2.getOutEdges().size() - o2.getInEdges().size();
			if(d1 > d2)				// because of the use of PriorityQueue, here the comparison is reverse
				return -1;
			else if (d1<d2)
				return 1;
			else
				return 0;
		}
	}

	//---------------------task properties used in ICPCP algorithm---------------------------
	private double EST = -1, EFT = -1, LFT = -1, AST = -1, AFT = -1;  //'-1' means the value has not been set
	private Task criticalParent;
	private boolean isAssigned = false;		//assigned以后EST就表示实际的开始时间了；EFT和LFT都设为   finish time，与论文不同

	public double getEST() {		return EST;	}
	public void setEST(double eST) {		EST = eST;	}
	public double getEFT() {		return EFT;	}
	public void setEFT(double eFT) {		EFT = eFT;	}
	public double getLFT() {		return LFT;	}
	public void setLFT(double lFT) {		LFT = lFT;	}
	public Task getCriticalParent() {		return criticalParent;	}
	public void setCriticalParent(Task criticalParent) {		this.criticalParent = criticalParent;	}
	public boolean isAssigned() {		return isAssigned;	}
	public void setAssigned(boolean isAssigned) {		this.isAssigned = isAssigned;	}
	public double getAST() {		return AST;	}
	public void setAST(double aST) {		AST = aST;	}
	public double getAFT() {		return AFT;	}
	public void setAFT(double aFT) {		AFT = aFT;	}
}