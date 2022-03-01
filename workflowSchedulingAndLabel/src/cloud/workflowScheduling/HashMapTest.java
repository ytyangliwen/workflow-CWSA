package cloud.workflowScheduling;
import java.io.IOException;
import java.util.*;

import cloud.workflowScheduling.setting.DeepCopyUtil;
import cloud.workflowScheduling.setting.VM;
class HashMapTest 
{
	public static void main(String[] args) throws ClassNotFoundException, IOException 
	{
		HashMap<VM,String> hm = new HashMap<VM,String>();

		//想要保证有序。存入的顺序和取出的顺序一致。
		//这时可以用一个在哈希表中加入链表结构的一个对象LinkedHashMap。该对象是HashMap的子类。
		hm = new LinkedHashMap<VM,String>();
		VM v1 = new VM(3);
		VM v2 = DeepCopyUtil.copy(v1);
		hm.put(new VM(0),"北京");
		hm.put(new VM(1),"上海");
		hm.put(v1,"广州");
		hm.put(v2,"天津");

//		hm.put(new Student("lisi4",24),"铁岭");


		Set<Map.Entry<VM,String>> entrySet = hm.entrySet();

		Iterator<Map.Entry<VM,String>> it = entrySet.iterator();

		while(it.hasNext())
		{
			Map.Entry<VM,String> me = it.next();

			VM stu = me.getKey();
			String add = me.getValue();

			System.out.println(stu.getId()+"..."+stu.getType()+"....."+add);
		}
		
//		HashMap<Student,String> hm = new HashMap<Student,String>();

//		//想要保证有序。存入的顺序和取出的顺序一致。
//		//这时可以用一个在哈希表中加入链表结构的一个对象LinkedHashMap。该对象是HashMap的子类。
//		hm = new LinkedHashMap<Student,String>();
//
//		hm.put(new Student("lisi1",21),"北京");
//		hm.put(new Student("lisi1",21),"上海");
//		hm.put(new Student("lisi2",22),"广州");
//		hm.put(new Student("lisi6",26),"天津");
////		hm.put(new Student("lisi4",24),"铁岭");
//
//
//		Set<Map.Entry<Student,String>> entrySet = hm.entrySet();
//
//		Iterator<Map.Entry<Student,String>> it = entrySet.iterator();
//
//		while(it.hasNext())
//		{
//			Map.Entry<Student,String> me = it.next();
//
//			Student stu = me.getKey();
//			String add = me.getValue();
//
//			System.out.println(stu.getName()+"..."+stu.getAge()+"....."+add);
//		}
	}
}

/*
要保证学生对象的唯一性。需要建立学生对象自身的判断相同的依据。
而且要根据学生的判断条件来定义依据。
因为是存放了Hash表中，所以要覆盖hashCode方法，和equals方法。
*/
class Student
{
	private String name;
	private int age;
	Student(String name,int age)
	{
		this.name = name;
		this.age = age;
	}
	public int hashCode()
	{
		final int NUM = 23;
		return name.hashCode()+age*NUM;
	}
	public boolean equals(Object obj)
	{
		if(this==obj)
			return true;
		if(!(obj instanceof Student))
			return false;
		Student stu = (Student)obj;
		return this.name.equals(stu.name) && this.age==stu.age;
	}
	public String getName()
	{
		return name;
	}
	public int getAge()
	{
		return age;
	}
}
