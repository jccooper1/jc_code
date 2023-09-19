
#include <stdio.h>
#include <stdlib.h>
#include <Macos.h>

int main()
{
    //打开Safari浏览器
    system("start safari");
    
    //等待一秒
    Sleep(1000);
    
    //模拟键盘输入百度网址
    keybd_event('W', 0, 0, 0); //按下W键
    keybd_event('W', 0, KEYEVENTF_KEYUP, 0); //松开W键
    keybd_event(VK_CONTROL, 0, 0, 0); //按下Ctrl键
    keybd_event('L', 0, 0, 0); //按下L键
    keybd_event('L', 0, KEYEVENTF_KEYUP, 0); //松开L键
    keybd_event(VK_CONTROL, 0 ,KEYEVENTF_KEYUP ,0); //松开Ctrl键
    
    Sleep(500);
    
    keybd_event(VK_BACK ,0 ,KEYEVENTF_EXTENDEDKEY ,0 );//按下Backspace键
	keybd_event(VK_BACK ,0 ,KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP ,0 );//松开Backspace键
	
	Sleep(500);
	
	keybd_event('H' ,MapVirtualKey ('H' ,MAPVK_VK_TO_VSC) ,KEYEVENTF_EXTENDEDKEY | KEYEVENTF_SCANCODE ,NULL );//按下H键
	keybd_event('H' ,MapVirtualKey ('H' ,MAPVK_VK_TO_VSC) ,KEYEVENTF_EXTENDEDKEY | KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP,NULL);//松开H键
	
	keybd_event('T' ,MapVirtualKey ('T' ,MAPVK_VK_TO_VSC) ,KEYEVENTF_EXTENDEDKEY | KEYEVENTF_SCANCODE,NULL);//按下T键
	keybd_event('T' ,MapVirtualKey ('T' ,MAPVK_VK_TO_VSC) ,
	KEYEVENTF_EXTENDEDKEY | KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP,NULL);//松开T键
	
	keybd_event('T' ,
	MapVirtualKey ('T' ,
	MAPVK_VK_TO_VSC) ,
	KEYEVENTF_EXTENDEDKEY |
	KEYEVENTF_SCANCODE,
	NULL);//按下T键
	
	keybd_event('T',
	MapVirtualKey ('T',
	MAPVK_VK_TO_VSC),
	KEYEVENTF_EXTENDEDKEY |
	KEYEVENTF_SCANCODE |
	KEYEVENTF_KEYUP,
	NULL);//松开T键
	
	keybd_event('P',
	MapVirtualKey ('P',
	MAPVK_VK_TO_VSC),
	KEYEVENTF_EXTENDEDKEY |
	KEYVENT
}