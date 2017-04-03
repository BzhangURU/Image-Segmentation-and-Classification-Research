// T02_filter.cpp
//Bodong Zhang

//This program helps to segment choromosomes in images and also find end points of chromosomes.
#include "stdafx.h"
#include <opencv.hpp>
#include"highgui.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>     // std::cout
#include <cstddef>      // std::size_t
#include <cmath>        // std::sqrt(double)
#include <queue>
#include <unordered_set>
using namespace cv;
using namespace std;
double RATE_LOW=0.1;//estimate the mean of background
double LARGE_STD=2;//if intensity<(background mean-large_STD*std), then set as chromosome
double  SMALL_STD=1;//if intensity<(background mean-SMALL_STD*std), then maybe it is chromosome
int RAD=3;
string pic_num_str="003";//003;010;015;014
void denoise_and_mark_region_index(Mat mask, Mat& mask_denoise,vector<vector<int>>& VofV_show_region_index,int& region_num);
//output: image with three labels(negative, positive, half positive)

void erosion(Mat& mask_denoise,vector<vector<int>>& VofV_show_region_index,Mat& mask_erosion,int peel_layer);

void visualize_end_points(Mat& mask_ero_denoise,vector<vector<pair<int,int>>>& end_points);
void get_end_points_process_one_region(vector<vector<int>>& VofV_show_ero_region_index,int seed_x,int seed_y,int region_ind,
																																											vector<vector<pair<int,int>>>& end_points);
void get_end_points(Mat& mask_ero_denoise,vector<vector<int>>& VofV_show_ero_region_index,
																																int ero_region_num,vector<vector<pair<int,int>>>& end_points);

void get_end_points(Mat& mask_ero_denoise,vector<vector<int>>& VofV_show_ero_region_index,
			int ero_region_num,vector<vector<pair<int,int>>>& end_points,vector<pair<int,int>>& seed_points,vector<pair<int,int>>& max_points);

void get_end_points_process_one_region(vector<vector<int>>& VofV_show_ero_region_index,int seed_x,int seed_y,int region_ind,
							vector<vector<pair<int,int>>>& end_points,vector<pair<int,int>>& max_points);

void visualize_end_points(Mat& mask_ero_denoise,vector<vector<pair<int,int>>>& end_points,vector<pair<int,int>>& seed_points,vector<pair<int,int>>& max_points);
int main( int argc, char** argv )
{
	
	Mat src, src_gray;
	Mat dst;
	string path= "E:\\U\\Image_Lab\\segmentation_code\\basic_segmentation\\T02_filter\\PB"+pic_num_str+".png";
	src = imread( path);
	cvtColor( src, src_gray, CV_BGR2GRAY );
	dst.create( src_gray.size(), src_gray.type() );
	blur( src_gray, dst, Size(3,3) );


	//check histogram
	int hist[256];
	for(int i=0;i<256;i++)
		hist[i]=0;
	//int intensity;
	for(int line_i=0;line_i<src_gray.rows;line_i++)
	{
		uchar* p_dst = dst.data+line_i*dst.cols;
		for(int column_j=0;column_j<dst.cols;column_j++)
		{
			hist[p_dst[column_j]]++;
		}
	}
	//get background mean and standard deviation///////////////////////////////////////////////////////////////////////
	double background_mean=0;
	double count=0;
	double count_background=0;
	double Total_N=(double)(dst.rows*dst.cols);
	for(int i=0;i<256;i++){
		count+=hist[i];
		if(count>Total_N*RATE_LOW){
			background_mean+=(double)(i*hist[i]);
			count_background+=hist[i];
		}
	}
	background_mean/=count_background;
	double background_std=0;
	for(int i=0;i<256;i++){
		count+=hist[i];
		if(count>Total_N*RATE_LOW){
			background_std+=(double)hist[i]*((double)i-background_mean)*((double)i-background_mean);
		}
	}
	background_std/=count_background;
	background_std=sqrt(background_std);
	background_std=MAX(10,background_std);//std=22.4
	
	Mat mask;
	mask.create( src_gray.size(), src_gray.type() );


	//in a square, if half of pixels is not background or sth(based on mean and standard deviation), then consider center as chromosome
	//three lables, negative, positive, half positive
	for(int line_i=0;line_i<dst.rows;line_i++)
	{
		uchar* p_dst = dst.data+line_i*dst.cols;
		uchar* p_mask= mask.data+line_i*mask.cols;
		for(int column_j=0;column_j<dst.cols;column_j++)
		{
			//hist[p_src_gray[column_j]]++;
			int total_in_box=0;int positive_in_box=0;int half_positive_in_box=0;
			for(int line_scan=(MAX(0,line_i-RAD));line_scan<MIN(dst.rows-1,line_i+RAD);line_scan++){
				uchar* p_dst_scan = dst.data+line_scan*dst.cols;
				for(int column_scan=(MAX(0,column_j-RAD));column_scan<MIN(dst.cols-1,column_j+RAD);column_scan++){
					total_in_box++;
					if((double)p_dst_scan[column_scan]<background_mean-background_std*LARGE_STD)
						positive_in_box++;
					if((double)p_dst_scan[column_scan]<background_mean-background_std*SMALL_STD)
						half_positive_in_box++;
				}
			}
			if(2*positive_in_box>=total_in_box)//@@@
				p_mask[column_j]=255;
			else if(2*half_positive_in_box>=total_in_box)//@@@
				p_mask[column_j]=128;
			else p_mask[column_j]=0;

		}
	}

	string str="E:\\U\\Image_Lab\\segmentation_code\\basic_segmentation\\T02_filter\\PB"+pic_num_str+"_try.png";
	if (imwrite(str,mask)==0)
	{
		printf("fail to save image\n");
		exit(0);
	}
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//if in an area(positive and half-positive), there is only(or above 80%) half-positive, then background
	//if in an area(positive and half-positive), the area is so small, then background
	//bool show_scan[][]=new bool[dst.rows][dst.cols];

	vector<vector<int>> VofV_show_region_index;
	int region_num;
	Mat mask_denoise;
	mask_denoise.create( src_gray.size(), src_gray.type() );
	denoise_and_mark_region_index(mask, mask_denoise,VofV_show_region_index,region_num);
	//!!!!!!!!!!!!!!!!!!!!!!!!another concern that has not be seen and has not been solved is that backgroud may appear "in" chromosome

	
	//erosion may cause region number increase(or decrease)!!!! So after erosion, maybe it's better to calculate region ind again
	Mat mask_erosion;
	mask_erosion.create( src_gray.size(), src_gray.type() );
	mask_denoise.copyTo(mask_erosion);
	erosion(mask_denoise,VofV_show_region_index,mask_erosion,3);//@@@@@@ 3, width of chromosome is 10


	//after erosion, there will exists new noise points(not connected to main large regions), 
	//also the number of regions change, so do "denoise_and_mark_region_index again"
	Mat mask_ero_denoise;
	mask_ero_denoise.create( src_gray.size(), src_gray.type() );
	vector<vector<int>> VofV_show_ero_region_index;
	int ero_region_num;
	denoise_and_mark_region_index(mask_erosion, mask_ero_denoise,VofV_show_ero_region_index,ero_region_num);


	//begin detecting end points
	vector<vector<pair<int,int>>> end_points;// end_points[7][2] means the second end point in region 7
	vector<pair<int,int>> seed_points;// seed_points[7] means the seedpoint in region 7
	vector<pair<int,int>> max_points;// seed_points[7] means the maxpoint in region 7
	//get_end_points(mask_ero_denoise,VofV_show_ero_region_index,ero_region_num,end_points);//get all end_point
	get_end_points(mask_ero_denoise,VofV_show_ero_region_index,ero_region_num,end_points,seed_points,max_points);//get all end_point


	//visualize_end_points(mask_ero_denoise,end_points);
	visualize_end_points(mask_ero_denoise,end_points,seed_points,max_points);

	//strech skeliton, then we are able to estimate width of chromosome

	//Future Plan:
	//if there is a "T" shape, then seperate them
	//if there is a "L" shape, then analyze it














	src.release();
	src_gray.release();
	dst.release();
	mask.release();
	mask_denoise.release();
	mask_erosion.release();
	mask_ero_denoise.release();
	return 0;
}


//input is an image with negetive(0), half-positive(128) and positive(255) 
//output is 1 an image with negetive(0), half-positive(128) and positive(255) (but no noise areas described below)
//					2 vector<vector<int>> same size as image, show the region index
//					3 number of positive regions(m)
//if in an area(positive and half-positive), there is only(or above 80%) half-positive, then set as background
//if in an area(positive and half-positive), the area(compare with medium) is so small, then set as background
//mark region index(0,1,2,3,4...m), there are totally m positive regions(based on half-positive), (the next step is try to segment them)
//ex:  0 2 0 0 0 1 0
//       2 2 2 0 0 1 0(4 neighbors connectivity, half-positive is considered as positive when segmenting regions)
//       0 0 0 0 3 0 0(no noise region, 0 is background,m=3 here)
void denoise_and_mark_region_index(Mat mask, Mat& mask_denoise,vector<vector<int>>& VofV_show_region_index,int& region_num){
	vector<vector<bool>> show_scan;
	//initialize
	VofV_show_region_index.clear();
	for(int i=0;i<mask.rows;i++){
		vector<bool> scan_v(mask.cols,false);
		vector<int> region_v(mask.cols,-1);
		//for(int j=0;j<mask.cols;j++){
			//scan_v.push_back(false);
			//region_v.push_back(-1);
		//}
		show_scan.push_back(scan_v);
		VofV_show_region_index.push_back(region_v);
	}
	//directly mark background as scanned
	for(int i=0;i<mask.rows;i++){
		uchar* p_mask = mask.data+i*mask.cols;
		uchar* p_mask_denoise = mask_denoise.data+i*mask_denoise.cols;
		for(int j=0;j<mask.cols;j++){
			if(p_mask[j]==0){
				show_scan[i][j]=true;
				p_mask_denoise[j]=0;
				VofV_show_region_index[i][j]=0;
			}
		}
	}

	//At first, mark index of all regions, including some noise regions except regions that there are almost no positive(just half-positive)
	vector<bool> keep_this_region;
	vector<int> region_area;
	int temp_region_num=0;
	//put a seed, grow it to calculate area and check the percentage of half positive
	for(int i=0;i<mask.rows;i++){
		uchar* p_mask = mask.data+i*mask.cols;
		for(int j=0;j<mask.cols;j++){
			//a new region candidate(would be a new region if it is not noise)
			if(!show_scan[i][j]){
				
				double count_area=0;
				double count_positive_num=0;
				queue<int> queue;
				unordered_set<int> dict;//!!!!!!!!!!!!!!!!!!!!!!future plan: can save dict for further process
				queue.push(i*mask.cols+j);//!!!!!!!!!!!!!!!!!!!! future plan: multiplication is time-consuming, use pair!!!
				show_scan[i][j]=true;
				//dict.insert(i*mask.cols+j);
				while(!queue.empty()){
					int index=queue.front();
					queue.pop();
					int line=index/mask.cols;
					int col=index%mask.cols;
					
					//show_scan[line][col]=true;
					//cout<<"\n";
					//cout<<"line= "<<line<<"     col= "<<col<<endl;
					//cout<<"show_scan= "<<show_scan[line][col]<<endl;
					dict.insert(line*mask.cols+col);
					count_area++;
					if((mask.data+line*mask.cols)[col]==255)
						count_positive_num++;
					//if(line>0&&!show_scan[line-1][col]&&(mask.data+(line-1)*mask.cols)[col]>100)
					if(line>0&&!show_scan[line-1][col]){//background is automatically set as scanned
						show_scan[line-1][col]=true;
						queue.push((line-1)*mask.cols+col);
					}
					if(line<mask.rows-1&&!show_scan[line+1][col]){
						//cout<<"put line= "<<line+1<<"  col= "<<col<<endl;
						show_scan[line+1][col]=true;
						queue.push((line+1)*mask.cols+col);
					}
					if(col>0&&!show_scan[line][col-1]){
						//cout<<"put line= "<<line<<"  col= "<<col-1<<endl;
						show_scan[line][col-1]=true;
						queue.push((line)*mask.cols+col-1);
					}
					if(col<mask.cols-1&&!show_scan[line][col+1]){
						//cout<<"put line= "<<line<<"  col= "<<col+1<<endl;
						show_scan[line][col+1]=true;
						queue.push((line)*mask.cols+col+1);
					}
					

				}
				//@@@
				if(count_positive_num>=0.2*count_area&&count_area>10){//add "&&count_area>10" to previously delete too small region
					temp_region_num++;
					keep_this_region.push_back(true);//wait for assessment, if area so small, delete it
					region_area.push_back(count_area);
					std::unordered_set<int>::const_iterator iter;
					for(iter=dict.begin();iter!=dict.end();iter++){
						int line=(*iter)/mask.cols;
						int col=(*iter)%mask.cols;
						VofV_show_region_index[line][col]=temp_region_num;
						*(mask_denoise.data+line*mask_denoise.cols+col)=*(mask.data+line*mask.cols+col);
					}
				}else{//region has no enough positive pixels, don't consider it as region
					std::unordered_set<int>::const_iterator iter;
					for(iter=dict.begin();iter!=dict.end();iter++){
						int line=(*iter)/mask.cols;
						int col=(*iter)%mask.cols;
						VofV_show_region_index[line][col]=0;
						*(mask_denoise.data+line*mask_denoise.cols+col)=0;
					}
				}
				dict.clear();//@@@
			}
		}
	}
	//sort area, use median as standard, delete any region whose area is too small
	vector<int> sort_area(region_area);
	sort(sort_area.begin(),sort_area.end());
	int medium_area=sort_area[sort_area.size()/2];
	//delete any region whose area is too small
	//use map to map new index
	//ex:    0 1 2 3 4 5 6   region 0 is background , region 2 is so small, so new region index is 0 1 0 2 3 4 5
	region_num=0;
	vector<int> map_region_ind;
	map_region_ind.push_back(0);
	for(int i=0;i<region_area.size();i++){
		//@@@
		if(10*region_area[i]<medium_area){
			keep_this_region[i]=false;
			map_region_ind.push_back(0);
		}else{
			region_num++;
			map_region_ind.push_back(region_num);
		}
	}
	for(int i=0;i<mask.rows;i++){
		//uchar* p_mask = mask.data+i*mask.cols;
		uchar* p_mask_denoise= mask_denoise.data+i*mask_denoise.cols;
		for(int j=0;j<mask.cols;j++){
			VofV_show_region_index[i][j]=map_region_ind[VofV_show_region_index[i][j]];
			if(VofV_show_region_index[i][j]==0)
				p_mask_denoise[j]=0;
		}
	}





	/////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//visualize VofV_show_region_index by imwrite image
	//can comment those code if you don't want to visualize image to write it to folder

	
	Mat mask_visual_color;
	cvtColor(mask,mask_visual_color,CV_GRAY2BGR);
	for(int i=0;i<mask.rows;i++){
		uchar* p_mask = mask.data+i*mask.cols;
		uchar* p_mask_visual_color= mask_visual_color.data+i*mask_visual_color.cols*mask_visual_color.channels();
		for(int j=0;j<mask.cols;j++){
			if(VofV_show_region_index[i][j]==0){
				p_mask_visual_color[3*j]=0;
				p_mask_visual_color[3*j+1]=0;
				p_mask_visual_color[3*j+2]=0;
			}else{
				p_mask_visual_color[3*j]=30*VofV_show_region_index[i][j];
				p_mask_visual_color[3*j+1]=250-17*VofV_show_region_index[i][j];
				p_mask_visual_color[3*j+2]=120-22*VofV_show_region_index[i][j];
			}
		}
	}
	string str1="E:\\U\\Image_Lab\\segmentation_code\\basic_segmentation\\T02_filter\\PB"+pic_num_str+"_mask_denoise.png";
	if (imwrite(str1,mask_denoise)==0)
	{
		printf("fail to save image\n");
		exit(0);
	}

	string str2="E:\\U\\Image_Lab\\segmentation_code\\basic_segmentation\\T02_filter\\PB"+pic_num_str+"_visualize.png";
	if (imwrite(str2,mask_visual_color)==0)
	{
		printf("fail to save image\n");
		exit(0);
	}
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////


}




//connection is 4-connection
//input: 1 mask_denoise:  an image with negetive(0), half-positive(128) and positive(255)
//				2 VofV_show_region_index, (has deleted all bad regions)refer to previous function's definition, more accurate
//				3 peel_layer: how many layers do you want to peel
//output: mask_erosion: an image with negetive(0), half-positive(128) and positive(255), after erosion
//!!!!!!!!!!!!before calling this function, mask_erosion has exactly same values with mask_denoise!!!!!!!!!!!
void erosion(Mat& mask_denoise,vector<vector<int>>& VofV_show_region_index,Mat& mask_erosion,int peel_layer){
	for(int i=0;i<mask_denoise.rows;i++){
		uchar* p_denoise = mask_denoise.data+i*mask_denoise.cols;
		uchar* p_erosion= mask_erosion.data+i*mask_erosion.cols;
		for(int j=0;j<mask_denoise.cols;j++){
			//if(VofV_show_region_index[i][j]!=0){
			if(p_denoise[j]!=0){
				for(int temp_i=max(0,i-peel_layer);temp_i<=min(mask_denoise.rows-1,i+peel_layer);temp_i++){
					for(int temp_j=max(0,j-peel_layer);temp_j<=min(mask_denoise.cols-1,j+peel_layer);temp_j++){
						if(abs(temp_i-i)+abs(temp_j-j)<=peel_layer)
							if(*(mask_denoise.data+temp_i*mask_denoise.cols+temp_j)==0){
								p_erosion[j]=0;
								break;
							}

					}
					if(p_erosion[j]==0)
						break;
				}
			}//else p_erosion[j]=0;
		}
	}



	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//visualize erosion
	string str_ero="E:\\U\\Image_Lab\\segmentation_code\\basic_segmentation\\T02_filter\\PB"+pic_num_str+"_mask_denoise_erosion.png";
	if (imwrite(str_ero,mask_erosion)==0)
	{
		printf("fail to save image\n");
		exit(0);
	}

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

}



//The function is to get end points for each region
//randomly select a point, calculate the distance between that point to all other pixels(the distance is measured by connection)
//if a pixel has local maximum(WxW neighbors has no pixel that has larger value than it, since width is 10, so based on that), then it is endpoint
//improvment: randomly choose a point(scan image and use first non_background pixel), all the detected end point should 
//be far away from that start point, then choose the second point as starter, that second point is far away from first point, then 
//end point detect region is only within neighbor of first point

//implementation: randomly choose a seed point(scan image and use first non_background pixel), set it as 1(region 1) 
//in show_region_index, then calculate distance between this pixel and all other pixels connect to it. Then scan the region again, 
//if they are far enough from seed, get local maximum. then use global maximum as second seed, calculate distance again, 
//only get local maximum from area that is close enough to first seed.

//input: 1: mask_ero_denoise: mask image that has been erosed and denoised with negetive(0), half-positive(128) and positive(255)
//				2: VofV_show_ero_region_index: same size as image, show the region index(0 is background, 1,2,3...ero_region_num are indexes)
//				3: ero_region_num: total number of regions in VofV_show_ero_region_index(check previous function introduction for details)
//output: vector<vector<pair<int,int>>> end_points;// end_points[7][2] means the (2+1)=3rd end point in region 7, !!!!!! end_points[0] is empty because
//!!!!!!!!!!!!!!!end_points[0] is empty because region 0 is background!!!!!!!!!!!!!!!!!!!!!!!!!!
void get_end_points(Mat& mask_ero_denoise,vector<vector<int>>& VofV_show_ero_region_index,
			int ero_region_num,vector<vector<pair<int,int>>>& end_points,vector<pair<int,int>>& seed_points,vector<pair<int,int>>& max_points){
	end_points.clear();
	seed_points.clear();
	max_points.clear();
	//initialize end_points
	for(int i=0;i<=ero_region_num;i++){
		vector<pair<int,int>> new_v;
		end_points.push_back(new_v);
		seed_points.push_back(make_pair(i*5,i*5));
		max_points.push_back(make_pair(i*5,i*3));
	}
	
	


	//show which region has been scanned and processed
	vector<bool> already_scan_region_ind(ero_region_num+1,false);

	for(int i=0;i<mask_ero_denoise.rows;i++){
		//uchar* p_ero_denoise = mask_ero_denoise.data+i*mask_ero_denoise.cols;
		for(int j=0;j<mask_ero_denoise.cols;j++){
			if(VofV_show_ero_region_index[i][j]!=0&&(!already_scan_region_ind[VofV_show_ero_region_index[i][j]])){///??????
				already_scan_region_ind[VofV_show_ero_region_index[i][j]]=true;
				seed_points[VofV_show_ero_region_index[i][j]].first=i;
				seed_points[VofV_show_ero_region_index[i][j]].second=j;
				get_end_points_process_one_region(VofV_show_ero_region_index,i,j,VofV_show_ero_region_index[i][j],end_points,max_points);

			}
		}
	}
			


}




//process a specific region, get end points of that region, then put them to end_points[region_ind],  region_ind>0(region_ind=0 is background)
//only output: end_points
void get_end_points_process_one_region(vector<vector<int>>& VofV_show_ero_region_index,int seed_x,int seed_y,int region_ind,
							vector<vector<pair<int,int>>>& end_points,vector<pair<int,int>>& max_points){
		
		//initialize all -1, only in region (region_ind) is not zero(seed is zero), non_zero number means distance to seed point(first seed point)
		//4 connectivity
		vector<vector<int>> temp_save_distance_to_seed;
		for(int row=0;row<VofV_show_ero_region_index.size();row++){
			vector<int> new_v(VofV_show_ero_region_index[0].size(),-1);
			temp_save_distance_to_seed.push_back(new_v);
		}
		int ROW=VofV_show_ero_region_index.size();
		int COL=VofV_show_ero_region_index[0].size();

		int max_dis=-1;
		int max_x=0,max_y=0;//location that has max_distance

		//grow that first seed point, calculate distance and save it to temp_save_distance_to_seed, in the meantime also get max_dis, max_x,max_y
		queue<pair<int,int>> queue;
		queue.push(make_pair(seed_x,seed_y));

		//hashset does not provide pair!!!!!!!!
		//unordered_set<pair<int,int>> dict;//save all points in that region
		vector<pair<int,int>> dict_v;//save all points in that region
		temp_save_distance_to_seed[seed_x][seed_y]=0;
		int count_dis=0;
		while(!queue.empty()){


			count_dis++;
			int queue_size=queue.size();


			for(int queue_ind=0;queue_ind<queue_size;queue_ind++){

				int point_x=queue.front().first;
				int point_y=queue.front().second;
				if(count_dis>max_dis){
					max_dis=count_dis;
					max_x=point_x;
					max_y=point_y;//this point will be the seond seed
				}
				queue.pop();
				//dict.insert(make_pair(point_x,point_y));
				dict_v.push_back(make_pair(point_x,point_y));
				//if [point_x-1][point_y] is not scanned and [point_x-1][point_y] is in the region
				if(point_x>0&&(temp_save_distance_to_seed[point_x-1][point_y]==-1)&&VofV_show_ero_region_index[point_x-1][point_y]==region_ind){
					temp_save_distance_to_seed[point_x-1][point_y]=count_dis;
					queue.push(make_pair(point_x-1,point_y));
				}
				if(point_x<ROW-1&&(temp_save_distance_to_seed[point_x+1][point_y]==-1)&&VofV_show_ero_region_index[point_x+1][point_y]==region_ind){
					temp_save_distance_to_seed[point_x+1][point_y]=count_dis;
					queue.push(make_pair(point_x+1,point_y));
				}
				if(point_y>0&&(temp_save_distance_to_seed[point_x][point_y-1]==-1)&&VofV_show_ero_region_index[point_x][point_y-1]==region_ind){
					temp_save_distance_to_seed[point_x][point_y-1]=count_dis;
					queue.push(make_pair(point_x,point_y-1));
				}
				if(point_y<COL-1&&(temp_save_distance_to_seed[point_x][point_y+1]==-1)&&VofV_show_ero_region_index[point_x][point_y+1]==region_ind){
					temp_save_distance_to_seed[point_x][point_y+1]=count_dis;
					queue.push(make_pair(point_x,point_y+1));
				}



			}





		}

		max_points[region_ind].first=max_x;
		max_points[region_ind].second=max_y;




		//find end points(far away from seed points) here
		//std::unordered_set<pair<int,int>>::const_iterator iter;
		//for(iter=dict.begin();iter!=dict.end();iter++){
		for(int dict_v_ind=0;dict_v_ind<dict_v.size();dict_v_ind++){
			//int point_x=(*iter).first;
			//int point_y=(*iter).second;
			int point_x=dict_v[dict_v_ind].first;
			int point_y=dict_v[dict_v_ind].second;
			//if far away from seed points   @@@@@@15@@@@@    future plan: this parameter is based on width of choromosome
			if(abs(point_x-seed_x)+abs(point_y-seed_y)>15){//!!!!!if you change this, you should also change same place 50 lines below!!!
				bool is_local_max=true;
				//check whether it is local maximum@@@@@@@@@@@@
				for(int search_x=max(0,point_x-6);search_x<=min(ROW,point_x+6);search_x++){//!!!!!if you change this, you should also change same place 50 lines below and 70 lines below!!!
					for(int search_y=max(0,point_y-6);search_y<=min(COL,point_y+6);search_y++){
						if(VofV_show_ero_region_index[search_x][search_y]==region_ind&&
																temp_save_distance_to_seed[point_x][point_y]<temp_save_distance_to_seed[search_x][search_y])
							is_local_max=false;

						if(!is_local_max)
							break;

					}
					if(!is_local_max)
							break;
				}
				if(is_local_max){
					end_points[region_ind].push_back(make_pair(point_x,point_y));
					//to avoid two same max num exist in neighbors, minus 1 to same max!!!!!!!!!!!!!!!!!!
					for(int search_x=max(0,point_x-6);search_x<=min(ROW,point_x+6);search_x++){////!!!!!if you change this, you should also change same place 10 lines above and 70 lines below!!!
						for(int search_y=max(0,point_y-6);search_y<=min(COL,point_y+6);search_y++){
							if(search_x!=point_x||search_y!=point_y)
								if(VofV_show_ero_region_index[search_x][search_y]==region_ind&&
																	temp_save_distance_to_seed[point_x][point_y]==temp_save_distance_to_seed[search_x][search_y])
									temp_save_distance_to_seed[search_x][search_y]--;
						}
					}
				}


			}
		}




		//////////////////////////
		//From here, start to find end points(close to seed points), calculate distance based on point(max_x,max_y)
		vector<vector<int>> temp_save_distance_to_max;
		for(int row=0;row<VofV_show_ero_region_index.size();row++){
			vector<int> new_v(VofV_show_ero_region_index[0].size(),-1);
			temp_save_distance_to_max.push_back(new_v);
		}
		//first get distance to max, also use queue

		/*comment this wrong part!
		//for(iter=dict.begin();iter!=dict.end();iter++){
		for(int dict_v_ind=0;dict_v_ind<dict_v.size();dict_v_ind++){
			//int point_x=(*iter).first;
			//int point_y=(*iter).second;
			int point_x=dict_v[dict_v_ind].first;
			int point_y=dict_v[dict_v_ind].second;
			temp_save_distance_to_max[point_x][point_y]=abs(point_x-max_x)+abs(point_y-max_y);
		}*/




		temp_save_distance_to_max[max_x][max_y]=0;
		while(!queue.empty())
			queue.pop();

		queue.push(make_pair(max_x,max_y));
		count_dis=0;

		while(!queue.empty()){


			count_dis++;
			int queue_size=queue.size();


			for(int queue_ind=0;queue_ind<queue_size;queue_ind++){

				int point_x=queue.front().first;
				int point_y=queue.front().second;
				queue.pop();
				//dict.insert(make_pair(point_x,point_y));
				//dict_v.push_back(make_pair(point_x,point_y));
				//if [point_x-1][point_y] is not scanned and [point_x-1][point_y] is in the region
				if(point_x>0&&(temp_save_distance_to_max[point_x-1][point_y]==-1)&&VofV_show_ero_region_index[point_x-1][point_y]==region_ind){
					temp_save_distance_to_max[point_x-1][point_y]=count_dis;
					queue.push(make_pair(point_x-1,point_y));
				}
				if(point_x<ROW-1&&(temp_save_distance_to_max[point_x+1][point_y]==-1)&&VofV_show_ero_region_index[point_x+1][point_y]==region_ind){
					temp_save_distance_to_max[point_x+1][point_y]=count_dis;
					queue.push(make_pair(point_x+1,point_y));
				}
				if(point_y>0&&(temp_save_distance_to_max[point_x][point_y-1]==-1)&&VofV_show_ero_region_index[point_x][point_y-1]==region_ind){
					temp_save_distance_to_max[point_x][point_y-1]=count_dis;
					queue.push(make_pair(point_x,point_y-1));
				}
				if(point_y<COL-1&&(temp_save_distance_to_max[point_x][point_y+1]==-1)&&VofV_show_ero_region_index[point_x][point_y+1]==region_ind){
					temp_save_distance_to_max[point_x][point_y+1]=count_dis;
					queue.push(make_pair(point_x,point_y+1));
				}



			}





		}





		//get end points(that close to seed points)
		//for(iter=dict.begin();iter!=dict.end();iter++){
		for(int dict_v_ind=0;dict_v_ind<dict_v.size();dict_v_ind++){
			//int point_x=(*iter).first;
			//int point_y=(*iter).second;
			int point_x=dict_v[dict_v_ind].first;
			int point_y=dict_v[dict_v_ind].second;
			
			//@@@@@@@@@@
			if(abs(point_x-seed_x)+abs(point_y-seed_y)<=15){////!!!!!if you change this, you should also change same place 50 lines above!!!
				bool is_local_max=true;
				//check whether it is local maximum@@@@@@@@@@@@
				for(int search_x=max(0,point_x-6);search_x<=min(ROW,point_x+6);search_x++){////!!!!!if you change this, you should also change same place 50 lines above and 20 lines below!!!
					for(int search_y=max(0,point_y-6);search_y<=min(COL,point_y+6);search_y++){
						//check if it is local maximum, how to handle two same max close to each other  ??? I solved it, check below
						if(VofV_show_ero_region_index[search_x][search_y]==region_ind&&
																temp_save_distance_to_max[point_x][point_y]<temp_save_distance_to_max[search_x][search_y])
							is_local_max=false;

						if(!is_local_max)
							break;

					}
					if(!is_local_max)
							break;
				}
				if(is_local_max){
					end_points[region_ind].push_back(make_pair(point_x,point_y));
					//to avoid two same max num exist in neighbors, minus 1 to same max!!!!!!!!!!!!!!!!!!
					for(int search_x=max(0,point_x-6);search_x<=min(ROW,point_x+6);search_x++){////!!!!!if you change this, you should also change same place 50 lines above and 70 lines above!!!
						for(int search_y=max(0,point_y-6);search_y<=min(COL,point_y+6);search_y++){
							if(search_x!=point_x||search_y!=point_y)
								if(VofV_show_ero_region_index[search_x][search_y]==region_ind&&
																	temp_save_distance_to_max[point_x][point_y]==temp_save_distance_to_max[search_x][search_y])
									temp_save_distance_to_max[search_x][search_y]--;
						}
					}

				}


			}


		}


}




void visualize_end_points(Mat& mask_ero_denoise,vector<vector<pair<int,int>>>& end_points,vector<pair<int,int>>& seed_points,vector<pair<int,int>>& max_points){
	Mat mask_visual_color_end_points;
	cvtColor(mask_ero_denoise,mask_visual_color_end_points,CV_GRAY2BGR);
	for(int i=0;i<mask_ero_denoise.rows;i++){
		uchar* p_mask = mask_ero_denoise.data+i*mask_ero_denoise.cols;
		uchar* p_mask_ends= mask_visual_color_end_points.data+i*mask_visual_color_end_points.cols*mask_visual_color_end_points.channels();
		for(int j=0;j<mask_ero_denoise.cols;j++){
			p_mask_ends[3*j]=p_mask[j];
			p_mask_ends[3*j+1]=p_mask[j];
			p_mask_ends[3*j+2]=p_mask[j];

		}
	}
	for(int region_ind=1;region_ind<end_points.size();region_ind++)
		for(int scan=0;scan<end_points[region_ind].size();scan++){
			int pt_x=end_points[region_ind][scan].first;
			int pt_y=end_points[region_ind][scan].second;


			for(int search_x=max(0,pt_x-1);search_x<=min(mask_ero_denoise.rows,pt_x+1);search_x++){
					for(int search_y=max(0,pt_y-1);search_y<=min(mask_ero_denoise.cols,pt_y+1);search_y++){
						uchar* p_mask_ends= mask_visual_color_end_points.data+search_x*mask_visual_color_end_points.cols*mask_visual_color_end_points.channels();
						p_mask_ends[3*search_y]=0;
						p_mask_ends[3*search_y+1]=0;
						p_mask_ends[3*search_y+2]=255;
					}
			}



		}

	for(int region_ind=1;region_ind<seed_points.size();region_ind++){
		int pt_x=seed_points[region_ind].first;
		int pt_y=seed_points[region_ind].second;

		uchar* p_mask_ends= mask_visual_color_end_points.data+pt_x*mask_visual_color_end_points.cols*mask_visual_color_end_points.channels();
		p_mask_ends[3*pt_y]=0;//Blue, max_point is blue
		p_mask_ends[3*pt_y+1]=255;//Green, seed point is green
		p_mask_ends[3*pt_y+2]=0;

		/*for(int search_x=max(0,pt_x-1);search_x<min(mask_ero_denoise.rows,pt_x+1);search_x++){
					for(int search_y=max(0,pt_y-1);search_y<min(mask_ero_denoise.cols,pt_y+1);search_y++){
						uchar* p_mask_ends= mask_visual_color_end_points.data+search_x*mask_visual_color_end_points.cols*mask_visual_color_end_points.channels();
						p_mask_ends[3*search_y]=0;//Blue
						p_mask_ends[3*search_y+1]=255;//Green, seed point is green
						p_mask_ends[3*search_y+2]=0;
					}
			}*/
	}

	for(int region_ind=1;region_ind<max_points.size();region_ind++){
		int pt_x=max_points[region_ind].first;
		int pt_y=max_points[region_ind].second;
		uchar* p_mask_ends= mask_visual_color_end_points.data+pt_x*mask_visual_color_end_points.cols*mask_visual_color_end_points.channels();
		p_mask_ends[3*pt_y]=255;//Blue, max_point is blue
		p_mask_ends[3*pt_y+1]=0;//Green, seed point is green
		p_mask_ends[3*pt_y+2]=0;
		/*for(int search_x=max(0,pt_x-1);search_x<min(mask_ero_denoise.rows,pt_x+1);search_x++){
					for(int search_y=max(0,pt_y-1);search_y<min(mask_ero_denoise.cols,pt_y+1);search_y++){
						uchar* p_mask_ends= mask_visual_color_end_points.data+search_x*mask_visual_color_end_points.cols*mask_visual_color_end_points.channels();
						p_mask_ends[3*search_y]=255;//Blue, max_point is blue
						p_mask_ends[3*search_y+1]=0;//Green, seed point is green
						p_mask_ends[3*search_y+2]=0;
					}
			}*/
	}

	string str_ends="E:\\U\\Image_Lab\\segmentation_code\\basic_segmentation\\T02_filter\\PB"+pic_num_str+"_end_points.png";
	if (imwrite(str_ends,mask_visual_color_end_points)==0)
	{
		printf("fail to save image\n");
		exit(0);
	}

}