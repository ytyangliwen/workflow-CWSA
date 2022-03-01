package cloud.workflowScheduling.setting;

import java.io.Serializable;

// virtual machine, i.e., cloud service resource
public class VM implements Serializable{

	public static final double LAUNCH_TIME = 0; //Amazon VM启动时间97s
	public static final long NETWORK_SPEED = 20 * 1024*1024; //贷款20M
	
	//4、文章中常用的, Minimizing文章, in the USA East region[US East(N. Virginia)]  General Purpose - Previous Generation
	//https://aws.amazon.com/ec2/previous-generation/?nc1=h_ls 
	//deadlineFactor 1.5, 2, 3, 4, ..., 9, 10   mpc用的这个
	public static final int TYPE_NO = 8;
////									m1.small, m1.medium, m3.medium, m1.large, m3.large, m1.xlarge, m3.xlarge, m3.2xlarge		
	public static final double[] SPEEDS = {1, 2, 3, 4, 6.5, 8, 13, 26,}; //ECU
	public static final double[] UNIT_COSTS = {0.044, 0.087, 0.067, 0.175, 0.133, 0.35, 0.266, 0.532}; //$
//	public static final double[] Bandwidth = {3.75, 7.5, 15, 30, 1.7, 3.75, 7.5, 15} //GB
	public static final double INTERVAL = 3600;	//one hour, billing interval
	public static final int FASTEST = 7;
	public static final int SLOWEST = 0;
	
	//1、L-ACO中的   subDMethods用的这个
//	public static final int TYPE_NO = 9;
//	public static final double[] SPEEDS = {1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5};
//	public static final double[] UNIT_COSTS = {0.12, 0.195, 0.28, 0.375, 0.48, 0.595, 0.72, 0.855, 1};
//	public static final double INTERVAL = 3600;	//one hour, billing interval
//	public static final int FASTEST = 8;
//	public static final int SLOWEST = 0;
	
	private static int internalId = 0;
	public static void resetInternalId(){	//called by the constructor of Solution
		internalId = 0;
	}
	public static void setInternalId(int startId){
		internalId = startId;
	}
	
	private int id;
	private int type; 

	public VM(int type){
		this.type = type;
		this.id = internalId++;
	}
	
//	public int hashCode()
//	{
//		final int NUM = 23;
//		return id*NUM;
//	}
//	public boolean equals(Object obj)
//	{
//		if(this==obj)
//			return true;
//		if(!(obj instanceof VM))
//			return false;
//		VM stu = (VM)obj;
//		return this.id==stu.id;
//	}
	//------------------------getters && setters---------------------------
	public void setType(int type) {		//can only be invoked in the same package, e.g., Solution
		this.type = type;
	}
	public void setId(int id) {
		this.id = id;
	}
	public double getSpeed(){		return SPEEDS[type];	}
	public double getUnitCost(){		return UNIT_COSTS[type];	}
	public int getId() {		return id;	}
	public int getType() {		return type;	}
	
	//-------------------------------------overrides--------------------------------
	public String toString() {
		return "VM [id=" + id + ", type=" + type + "]";
	}
}