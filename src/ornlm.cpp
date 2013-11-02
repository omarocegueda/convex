/*! \file ornlm.cpp
	\author Omar Ocegueda
	\brief Optimized (Blockwise) Rician-corrected Nonlocal Means filter. Addapted from Pierrick's code with some low level optimizations
*/
#include "ornlm.h"
#include "math.h"
#include "string.h"
#ifndef SQR
#define SQR(x) ((x)*(x))
#endif
/*! \brief Accumulate the values of one block (Rician model)
*/
void accumulateBlock(double *image, int nslices, int nrows, int ncols, int s, int r, int c, int patchLen, double weight, double *average){
	int sliceSize=nrows*ncols;
	int count=0;
	for(int k=-patchLen; k<=patchLen; ++k){
		for(int i=-patchLen; i<=patchLen; ++i){
			for(int j=-patchLen; j<=patchLen; ++j){
				int ss=s+k;
				int rr=r+i;
				int cc=c+j;
				bool outside=false;
				if((ss<0)||(ss>=nslices)){
					outside = true;
				}else if((rr<0)||(rr>=nrows)){
					outside = true;
				}else if((cc<0)||(cc>=ncols)){
					outside = true;
				}
				if (outside){
					int index=s*sliceSize+r*ncols+c;
					average[count]+=weight*SQR(image[index]);
				}else{
					int index=ss*sliceSize+rr*ncols+cc;
					average[count]+=weight*SQR(image[index]);
				}
				++count;
			}
		}
	}
}

/*! \brief Updates a block with its estimated value given by average
*/
void blockValue(double *estimate, double *label, int nslices, int nrows, int ncols, int s, int r, int c, int neighSize,double *average, double globalSum, double hh){
	int sliceSize=nrows*ncols;
	int count=0;
	for(int k=-neighSize; k<=neighSize; ++k){
		for(int i=-neighSize; i<=neighSize; ++i){
			for(int j=-neighSize; j<=neighSize; ++j, ++count){
				int ss=s+k;
				if((ss<0)||(ss>=nslices)){
					continue;
				}
				int rr=r+i;
				if((rr<0)||(rr>=nrows)){
					continue;
				}
				int cc=c+j;
				if((cc<0)||(cc>=ncols)){
					continue;
				}
				int index=ss*sliceSize+rr*ncols+cc;
				double denoised=average[count]/globalSum-hh;
				if (denoised>0){
					denoised=sqrt(denoised);
				}else{
					denoised=0.0;
				}
				estimate[index]+=denoised;
				label[index]+=1;
			}
		}
	}
}

/*! \brief Computes the distance between the patches centered at (c, r, s) and (nc, nr, ns) respectively. 
	The boundary of the image is given by ss, sr, sc and the size of the patch is (2*f+1)^3
*/
double patchDistance(double* image, int s, int r, int c, int ns, int nr, int nc, int patchLen, int nslices, int nrows, int ncols){
	double total=0;
	int sliceSize=nrows*ncols;
	for(int k=-patchLen;k<=patchLen; ++k){
		for(int i=-patchLen;i<=patchLen; ++i){
			for(int j=-patchLen;j<=patchLen; ++j){
				int nk1=s+k;
				int ni1=r+i;
				int nj1=c+j;
				int nk2=ns+k;
				int ni2=nr+i;
				int nj2=nc+j;
				if(ni1<0) ni1=-ni1;
				if(ni2<0) ni2=-ni2;
				if(nj1<0) nj1=-nj1;
				if(nj2<0) nj2=-nj2;
				if(nk1<0) nk1=-nk1;
				if(nk2<0) nk2=-nk2;
				if(ni1>=nrows) ni1=2*nrows-ni1-1;
				if(nj1>=ncols) nj1=2*ncols-nj1-1;
				if(nk1>=nslices) nk1=2*nslices-nk1-1;
				if(ni2>=nrows) ni2=2*nrows-ni2-1;
				if(nj2>=ncols) nj2=2*ncols-nj2-1;
				if(nk2>=nslices) nk2=2*nslices-nk2-1;
				int index1=nk1*sliceSize+ni1*ncols+nj1;
				int index2=nk2*sliceSize+ni2*ncols+nj2;
				total+=SQR(image[index1]-image[index2]);
			}
		}
	}
	return total/((2*patchLen+1)*(2*patchLen+1)*(2*patchLen+1));
}

/*! \brief Computes local mean and variance at each voxel using the 26-neighborhood voxel values as samples
*/
void computeLocalStatistics(double *image, int nslices, int nrows, int ncols, double *means, double *variances){
	int pos=0;
	int sliceSize=nrows*ncols;
	for(int s=0;s<nslices;++s){
		for(int r=0;r<nrows;++r){
			for(int c=0;c<ncols;++c, ++pos){
				double mean=0;
				for(int k=-1;k<=1;++k){
					for(int i=-1;i<=1;++i){
						for(int j=-1;j<=1;++j){
							int ss=s+k;
							int rr=r+i;
							int cc=c+j;
							if(ss<0){ss=-ss;}//FIXME
							if(rr<0){rr=-rr;}
							if(cc<0){cc=-cc;}
							if(ss>=nslices){ss=2*nslices-ss-1;}
							if(rr>=nrows){rr=2*nrows-rr-1;}
							if(cc>=ncols){cc=2*ncols-cc-1;}
							int neigh=ss*sliceSize+rr*ncols+cc;
							mean+=image[neigh];
						}
					}
				}
				mean/=27;
				means[pos]=mean;
				double var=0;
				int cnt=0;
				for(int k=-1;k<=1;++k){
					for(int i=-1;i<=1;++i){
						for(int j=-1;j<=1;++j){
							int ss=s+k;
							if((ss<=0)||(ss>=nslices)){continue;}//FIXME: "<="
							int rr=r+i;
							if((rr<0)||(rr>=nrows)){continue;}
							int cc=c+j;
							if((cc<0)||(cc>=ncols)){continue;}
							++cnt;
							int neigh=ss*sliceSize+rr*ncols+cc;
							var+=SQR(image[neigh]-mean);
						}
					}
				}
				variances[pos]=var/(cnt-1);
			}
		}
	}
}
/*! \brief The main function
*/
void ornlm(double *image, int nslices, int nrows, int ncols, int searchVolumeLen, int patchLen, double h, double *filtered){
	double hh=2*h*h;
	int nvox=nslices*nrows*ncols;
	int patchSize=(2*patchLen+1)*(2*patchLen+1)*(2*patchLen+1);
	double *average=new double[patchSize];
	double *means=new double[nvox];
	double *variances=new double[nvox];
	double *estimate=new double[nvox];
	double *label=new double[nvox];
	memset(estimate, 0, sizeof(double)*nvox);
	memset(label, 0, sizeof(double)*nvox);
	computeLocalStatistics(image, nslices, nrows, ncols, means, variances);
	//--filter--
	double epsilon	=1e-5;
	double mu1		=0.95;
	double var1		=0.5;
	int sliceSize=nrows*ncols;
	for(int s=0;s<nslices;s+=2){
		for(int r=0;r<nrows;r+=2){
			for(int c=0;c<ncols;c+=2){
				int current=s*sliceSize+r*ncols+c;
				memset(average, 0, sizeof(double)*patchSize);
				if((means[current]>epsilon)&&(variances[current]>epsilon)){//process this patch
					double wmax=0;
					double totalWeight=0;
					for(int k=-searchVolumeLen;k<=searchVolumeLen;++k){
						for(int i=-searchVolumeLen;i<=searchVolumeLen;++i){
							for(int j=-searchVolumeLen;j<=searchVolumeLen;++j){
								if((k==0)&&(i==0)&&(j==0)){
									continue;
								}
								int ss=s+k;
								if((ss<0)||(ss>=nslices)){continue;}
								int rr=r+i;
								if((rr<0)||(rr>=nrows)){continue;}
								int cc=c+j;
								if((cc<0)||(cc>=ncols)){continue;}
								int neigh=ss*sliceSize+rr*ncols+cc;
								if((means[neigh]<=epsilon)||(variances[neigh]<=epsilon)){
									continue;
								}
								if((means[current]<=mu1*means[neigh]) || (means[neigh]<=mu1*means[current])){//the means are not similar
									continue;
								}
								if((variances[current]<=var1*variances[neigh]) || (variances[neigh]<=var1*variances[current])){//the variances are not similar
									continue;
								}
								double d=patchDistance(image,s,r,c,ss,rr,cc,patchLen,nslices,nrows,ncols);
								double w=exp(-d/(h*h));
								if(w>wmax){
									wmax=w;
								}
								accumulateBlock(image, nslices, nrows, ncols, ss, rr, cc, patchLen, w, average);
								totalWeight+=w;
							}
						}
					}
					if(wmax==0.0){//FIXME
						wmax=1.0;
					}
					totalWeight+=wmax;
					accumulateBlock(image, nslices, nrows, ncols, s, r, c, patchLen, wmax, average);//accumulate current block with the maximum weight
					blockValue(estimate, label, nslices, nrows, ncols, s, r, c, patchLen, average, totalWeight, hh);
				}else{//((means[pos]<=epsilon)||(variances[pos]<=epsilon)){
					double totalweight=1.0;
					accumulateBlock(image, nslices, nrows, ncols, s, r, c, patchLen, totalweight, average);
					blockValue(estimate, label, nslices, nrows, ncols, s, r, c, patchLen, average, totalweight, hh);
				}
			}
		}
	}

	for(int voxel=0;voxel<nvox;++voxel){
		if(label[voxel]>0){
			filtered[voxel]=estimate[voxel]/label[voxel];
		}else{
			filtered[voxel]=image[voxel];
		}
	}
	
	delete[] average;
	delete[] means;
	delete[] variances;
	delete[] estimate;
	delete[] label;
}
