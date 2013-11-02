#include "TotalVariation.h"
#include <iostream>
#include <string.h>
#include <math.h>
#include "derivatives.h"
#include "macros.h"
#include "bits.h"
using namespace std;
#ifndef MAX
#define MAX(a,b) (((a)<(b))?(b):(a))
#endif

void filterTotalVariation_L2(double *g, int nrows, int ncols, double lambda, double tau, double sigma, double theta, double *x, void (*callback)(double*, int, int)){
	double *xbar=new double[nrows*ncols];
	double *yr=new double[nrows*ncols];
	double *yc=new double[nrows*ncols];
	double *dxdr=new double[nrows*ncols];
	double *dxdc=new double[nrows*ncols];
	double *divergence=new double[nrows*ncols];
	double tolerance=1e-9;
	double error=1+tolerance;
	int maxIter=5000;
	int iter=0;
	memset(yr, 0, sizeof(double)*nrows*ncols);
	memset(yc, 0, sizeof(double)*nrows*ncols);
	memcpy(xbar, x, sizeof(double)*nrows*ncols);
	while((tolerance<error) && (iter<=maxIter)){
		++iter;
		error=0;
		//update dual field
		computeGradient(xbar, nrows, ncols, dxdr, dxdc);
		for(int i=0;i<nrows;++i){
			for(int j=0;j<ncols;++j){
				yr[i*ncols+j]+=sigma*dxdr[i*ncols+j];
				yc[i*ncols+j]+=sigma*dxdc[i*ncols+j];
				double nrm=sqrt(yr[i*ncols+j]*yr[i*ncols+j] + yc[i*ncols+j]*yc[i*ncols+j]);
				if(nrm>1){
					yr[i*ncols+j]/=nrm;
					yc[i*ncols+j]/=nrm;
				}
			}
		}
		//update primal field
		computeDivergence(yr, yc, nrows, ncols, divergence);
		for(int i=0;i<nrows;++i){
			for(int j=0;j<ncols;++j){
				double diff=-x[i*ncols+j];
				x[i*ncols+j]+=tau*divergence[i*ncols+j];
				x[i*ncols+j]=(x[i*ncols+j]+tau*lambda*g[i*ncols+j])/(1.0+tau*lambda);
				diff+=x[i*ncols+j];
				error+=diff*diff;
				//update xbar
				xbar[i*ncols+j]=x[i*ncols+j]+theta*diff;
			}
		}
		error/=(nrows*ncols);
	}
	delete[] xbar;
	delete[] yr;
	delete[] yc;
	delete[] dxdr;
	delete[] dxdc;
	delete[] divergence;
}

void filterTotalVariation_L2(double *g, int nslices, int nrows, int ncols, double lambda, double tau, double sigma, double theta, double *x, void (*callback)(double*, int, int)){
	int nvox=nslices*nrows*ncols;
	double *xbar=new double[nvox];
	double *ys=new double[nvox];
	double *yr=new double[nvox];
	double *yc=new double[nvox];
	double *dxds=new double[nvox];
	double *dxdr=new double[nvox];
	double *dxdc=new double[nvox];
	double *divergence=new double[nvox];
	double tolerance=1e-9;
	double error=1+tolerance;
	int maxIter=5000;
	int iter=0;
	memset(ys, 0, sizeof(double)*nvox);
	memset(yr, 0, sizeof(double)*nvox);
	memset(yc, 0, sizeof(double)*nvox);
	memcpy(xbar, x, sizeof(double)*nvox);
	while((tolerance<error) && (iter<=maxIter)){
		++iter;
		error=0;
		//update dual field
		computeGradient(xbar, nslices, nrows, ncols, dxds, dxdr, dxdc);
		int pos=0;
		for(int s=0;s<nslices;++s){
			for(int i=0;i<nrows;++i){
				for(int j=0;j<ncols;++j, ++pos){
					ys[pos]+=sigma*dxds[pos];
					yr[pos]+=sigma*dxdr[pos];
					yc[pos]+=sigma*dxdc[pos];
					double nrm=ys[pos]*ys[pos] + yr[pos]*yr[pos] + yc[pos]*yc[pos];
					if(nrm>1){
						nrm=sqrt(nrm);
						ys[pos]/=nrm;
						yr[pos]/=nrm;
						yc[pos]/=nrm;
					}
				}
			}
		}
		//update primal field
		computeDivergence(ys, yr, yc, nslices, nrows, ncols, divergence);
		pos=0;
		error=0;
		for(int s=0;s<nslices;++s){
			for(int i=0;i<nrows;++i){
				for(int j=0;j<ncols;++j, ++pos){
					double diff=-x[pos];
					x[pos]+=tau*divergence[pos];
					x[pos]=(x[pos]+tau*lambda*g[pos])/(1.0+tau*lambda);
					diff+=x[pos];
					error+=diff*diff;
					//update xbar
					xbar[pos]=x[pos]+theta*diff;
				}
			}
		}
		error/=(nvox);
		cerr<<iter<<"/"<<maxIter<<": "<<error<<endl;
	}
	delete[] xbar;
	delete[] ys;
	delete[] yr;
	delete[] yc;
	delete[] dxds;
	delete[] dxdr;
	delete[] dxdc;
	delete[] divergence;
}

void filterTotalVariation_L1(double *g, int nrows, int ncols, double lambda, double tau, double sigma, double theta, double *x, void (*callback)(double*, int, int)){
	double *xbar=new double[nrows*ncols];
	double *yr=new double[nrows*ncols];
	double *yc=new double[nrows*ncols];
	double *dxdr=new double[nrows*ncols];
	double *dxdc=new double[nrows*ncols];
	double *divergence=new double[nrows*ncols];
	double tolerance=1e-5;
	double error=1+tolerance;
	int maxIter=5000;
	int iter=0;
	memset(yr, 0, sizeof(double)*nrows*ncols);
	memset(yc, 0, sizeof(double)*nrows*ncols);
	memcpy(xbar, x, sizeof(double)*nrows*ncols);
	while((tolerance<error) && (iter<=maxIter)){
		++iter;
		error=0;
		//update dual field
		computeGradient(xbar, nrows, ncols, dxdr, dxdc);
		for(int i=0;i<nrows;++i){
			for(int j=0;j<ncols;++j){
				yr[i*ncols+j]+=sigma*dxdr[i*ncols+j];
				yc[i*ncols+j]+=sigma*dxdc[i*ncols+j];
				//double nrm=sqrt(yr[i*ncols+j]*yr[i*ncols+j] + yc[i*ncols+j]*yc[i*ncols+j]);
				double nrm=MAX(fabs(yr[i*ncols+j]), fabs(yc[i*ncols+j]));
				if(nrm<1){
					nrm=1;
				}
				yr[i*ncols+j]/=nrm;
				yc[i*ncols+j]/=nrm;
			}
		}
		//update primal field
		computeDivergence(yr, yc, nrows, ncols, divergence);
		for(int i=0;i<nrows;++i){
			for(int j=0;j<ncols;++j){
				double diff=-x[i*ncols+j];
				double arg=x[i*ncols+j]+tau*divergence[i*ncols+j];
				double obs=g[i*ncols+j];
				if((arg-obs)>(tau*lambda)){
					x[i*ncols+j]=arg-tau*lambda;
				}else if((arg-obs)<-(tau*lambda)){
					x[i*ncols+j]=arg+tau*lambda;
				}else{
					x[i*ncols+j]=obs;
				}
				diff+=x[i*ncols+j];
				error+=diff*diff;
				//update xbar
				xbar[i*ncols+j]=x[i*ncols+j]+theta*diff;
			}
		}
		error/=(nrows*ncols);
		/*if((callback!=NULL) && (iter%20==0)){
			callback(x, nrows, ncols);
		}*/
		
	}
	delete[] xbar;
	delete[] yr;
	delete[] yc;
	delete[] dxdr;
	delete[] dxdc;
	delete[] divergence;
}


void filterHuber_L2(double *g, int nrows, int ncols, double alpha, double lambda, double tau, double sigma, double theta, double *x, void (*callback)(double*, int, int)){
	double *xbar=new double[nrows*ncols];
	double *yr=new double[nrows*ncols];
	double *yc=new double[nrows*ncols];
	double *dxdr=new double[nrows*ncols];
	double *dxdc=new double[nrows*ncols];
	double *divergence=new double[nrows*ncols];
	double tolerance=1e-4;
	double error=1+tolerance;
	int maxIter=5000;
	int iter=0;
	memset(yr, 0, sizeof(double)*nrows*ncols);
	memset(yc, 0, sizeof(double)*nrows*ncols);
	memcpy(xbar, x, sizeof(double)*nrows*ncols);
	while((tolerance<error) && (iter<=maxIter)){
		++iter;
		error=0;
		//update dual field
		computeGradient(xbar, nrows, ncols, dxdr, dxdc);
		for(int i=0;i<nrows;++i){
			for(int j=0;j<ncols;++j){
				yr[i*ncols+j]+=sigma*dxdr[i*ncols+j];
				yc[i*ncols+j]+=sigma*dxdc[i*ncols+j];
				yr[i*ncols+j]/=(1+sigma*alpha);
				yc[i*ncols+j]/=(1+sigma*alpha);
				double nrm=sqrt(yr[i*ncols+j]*yr[i*ncols+j] + yc[i*ncols+j]*yc[i*ncols+j]);
				if(nrm<1){
					nrm=1;
				}
				yr[i*ncols+j]/=nrm;
				yc[i*ncols+j]/=nrm;
			}
		}
		//update primal field
		computeDivergence(yr, yc, nrows, ncols, divergence);
		for(int i=0;i<nrows;++i){
			for(int j=0;j<ncols;++j){
				double diff=-x[i*ncols+j];
				x[i*ncols+j]+=tau*divergence[i*ncols+j];
				x[i*ncols+j]=(x[i*ncols+j]+tau*lambda*g[i*ncols+j])/(1.0+tau*lambda);
				diff+=x[i*ncols+j];
				error+=diff*diff;
				//update xbar
				xbar[i*ncols+j]=x[i*ncols+j]+theta*diff;
			}
		}
		error/=(nrows*ncols);
		/*if((callback!=NULL) && (iter%20==0)){
			callback(x, nrows, ncols);
		}*/
		
	}
	delete[] xbar;
	delete[] yr;
	delete[] yc;
	delete[] dxdr;
	delete[] dxdc;
	delete[] divergence;
}

void filterTGV_L2(double *g, int nrows, int ncols, double lambda, double alpha0, double alpha1, double tau, double sigma, double theta, double *u, double *v){
	//---primal variables---
	double *dubar_dr=new double[nrows*ncols];
	double *dubar_dc=new double[nrows*ncols];
	double *ubar=new double[nrows*ncols];
	double *vr=new double[nrows*ncols];
	double *vc=new double[nrows*ncols];
	double *vr_bar=new double[nrows*ncols];
	double *vc_bar=new double[nrows*ncols];
	double *dvr_bar_dr=new double[nrows*ncols];
	double *dvr_bar_dc=new double[nrows*ncols];
	double *dvc_bar_dr=new double[nrows*ncols];
	double *dvc_bar_dc=new double[nrows*ncols];
	
	memset(ubar, 0, sizeof(double)*nrows*ncols);
	memset(vr, 0, sizeof(double)*nrows*ncols);
	memset(vc, 0, sizeof(double)*nrows*ncols);
	memset(vr_bar, 0, sizeof(double)*nrows*ncols);
	memset(vc_bar, 0, sizeof(double)*nrows*ncols);
	memset(dvr_bar_dr, 0, sizeof(double)*nrows*ncols);
	memset(dvr_bar_dc, 0, sizeof(double)*nrows*ncols);
	memset(dvc_bar_dr, 0, sizeof(double)*nrows*ncols);
	memset(dvc_bar_dc, 0, sizeof(double)*nrows*ncols);
	//---dual variables-----
	double *pr=new double[nrows*ncols];
	double *pc=new double[nrows*ncols];
	double *qrr=new double[nrows*ncols];
	double *qcc=new double[nrows*ncols];
	double *qrc=new double[nrows*ncols];

	memset(pr, 0, sizeof(double)*nrows*ncols);
	memset(pc, 0, sizeof(double)*nrows*ncols);
	memset(qrr, 0, sizeof(double)*nrows*ncols);
	memset(qcc, 0, sizeof(double)*nrows*ncols);
	memset(qrc, 0, sizeof(double)*nrows*ncols);
	//-----------------
	double *divP=new double[nrows*ncols];
	double *divQr=new double[nrows*ncols];
	double *divQc=new double[nrows*ncols];
	double tolerance=1e-9;
	double error=1+tolerance;
	int maxIter=500;
	int iter=0;
	
	while((tolerance<error) && (iter<=maxIter)){
		++iter;
		error=0;
		//update dual field
		computeGradient(ubar, nrows, ncols, dubar_dr, dubar_dc);
		computeGradient(vr_bar, nrows, ncols, dvr_bar_dr, dvr_bar_dc);
		computeGradient(vc_bar, nrows, ncols, dvc_bar_dr, dvc_bar_dc);
		int pos=0;
		for(int i=0;i<nrows;++i){
			for(int j=0;j<ncols;++j, ++pos){
				//--update [pr,pc]--
				pr[pos]+=sigma*(dubar_dr[pos]-vr[pos]);
				pc[pos]+=sigma*(dubar_dc[pos]-vc[pos]);
				double nrm=pr[pos]*pr[pos]+pc[pos]*pc[pos];
				nrm=sqrt(nrm)/alpha0;
				if(nrm>1){
					pr[i*ncols+j]/=nrm;
					pc[i*ncols+j]/=nrm;
				}
				//--update [qrr,qrc,qcc]--
				qrr[pos]+=sigma*(dvr_bar_dr[pos]);
				qcc[pos]+=sigma*(dvc_bar_dc[pos]);
				qrc[pos]+=sigma*0.5*(dvr_bar_dc[pos]+dvc_bar_dr[pos]);
				nrm=qrr[pos]*qrr[pos] + qcc[pos]*qcc[pos] + 2.0*qrc[pos]*qrc[pos];
				nrm=sqrt(nrm)/alpha1;
				if(nrm>1){
					qrr[pos]/=nrm;
					qcc[pos]/=nrm;
					qrc[pos]/=nrm;
				}
				//------------------------
			}
		}
		//update primal field
		computeDivergence(pr, pc, nrows, ncols, divP);
		computeDivergence(qrr, qrc, nrows, ncols, divQr);
		computeDivergence(qrc, qcc, nrows, ncols, divQc);
		pos=0;
		for(int i=0;i<nrows;++i){
			for(int j=0;j<ncols;++j, ++pos){
				//--update u--
				double diff=-u[i*ncols+j];
				u[pos]+=tau*divP[pos];
				u[pos]=(u[pos]+tau*lambda*g[pos])/(1.0+tau*lambda);
				//update ubar
				diff+=u[pos];
				error+=diff*diff;
				ubar[pos]=u[pos]+theta*diff;
				//--update v--
				double diff_vr=-vr[pos];
				double diff_vc=-vc[pos];
				vr[i*ncols+j]+=tau*(pr[pos]+divQr[pos]);
				vc[i*ncols+j]+=tau*(pc[pos]+divQc[pos]);
				//update vbar
				diff_vr+=vr[pos];
				diff_vc+=vc[pos];
				vr_bar[pos]=vr[pos]+theta*diff_vr;
				vc_bar[pos]=vc[pos]+theta*diff_vc;
				//------------
			}
		}
	}
	if(v!=NULL){
		memcpy(v, vr, sizeof(double)*nrows*ncols);
		memcpy(&v[nrows*ncols], vc, sizeof(double)*nrows*ncols);
	}
	delete[] dubar_dr;
	delete[] dubar_dc;
	delete[] ubar;
	delete[] vr;
	delete[] vc;
	delete[] vr_bar;
	delete[] vc_bar;
	delete[] dvr_bar_dr;
	delete[] dvr_bar_dc;
	delete[] dvc_bar_dr;
	delete[] dvc_bar_dc;
	delete[] pr;
	delete[] pc;
	delete[] qrr;
	delete[] qcc;
	delete[] qrc;
	delete[] divP;
	delete[] divQr;
	delete[] divQc;
}

//===========================================
void unwrapTotalVariation(double *dgdr, double *dgdc, int nrows, int ncols, double lambda, double tau, double sigma, double theta, double *x, void (*callback)(double*, int, int)){
	double *xbar=new double[nrows*ncols];
	double *yr=new double[nrows*ncols];
	double *yc=new double[nrows*ncols];
	double *dxdr=new double[nrows*ncols];
	double *dxdc=new double[nrows*ncols];
	double *divergence=new double[nrows*ncols];
	double tolerance=1e-5;
	double error=1+tolerance;
	int maxIter=10000;
	int iter=0;
	memset(yr, 0, sizeof(double)*nrows*ncols);
	memset(yc, 0, sizeof(double)*nrows*ncols);
	memcpy(xbar, x, sizeof(double)*nrows*ncols);
	double errorPrev=1e10;
	while((tolerance<error) && (iter<=maxIter)){
		++iter;
		/*error=x[0]*x[0]*0.5;
		computeGradient(x, nrows, ncols, dxdr, dxdc, EDT_Forward, EBC_Circular);
		int pos=0;
		for(int i=0;i<nrows;++i){
			for(int j=0;j<ncols;++j, ++pos){
				error+=sqrt(SQR(dxdr[pos]-dgdr[pos])+SQR(dxdc[pos]-dgdc[pos]));
			}
		}
		error=sqrt(error/(nrows*ncols));
		cerr<<error<<endl;*/
		/*if((iter%1000)==0){
			cerr<<iter<<": "<<error<<endl;
			if(errorPrev<error){
				break;
			}
			errorPrev=error;
		}*/
		//update dual field
		computeGradient(xbar, nrows, ncols, dxdr, dxdc, EDT_Forward, EBC_Circular);
		//int pos=0;
		error=0;
		int pos=0;
		for(int i=0;i<nrows;++i){
			for(int j=0;j<ncols;++j, ++pos){
				double prevGrad[2]={yr[pos], yc[pos]};
				yr[pos]+=sigma*(dxdr[pos]-dgdr[pos]);
				yc[pos]+=sigma*(dxdc[pos]-dgdc[pos]);

				//----L1-L2---------------
				double nrm=yr[pos]*yr[pos] + yc[pos]*yc[pos];
				if(nrm>1){
					nrm=sqrt(nrm);
					yr[pos]/=nrm;
					yc[pos]/=nrm;
				}
				//----L1-L1---------------
				/*if(yr[pos]>1){
					yr[pos]=1;
				}else if(yr[pos]<-1){
					yr[pos]=-1;
				}
				if(yc[pos]>1){
					yc[pos]=1;
				}else if(yc[pos]<-1){
					yc[pos]=-1;
				}*/
				//----L2^2---------------
				/*yr[pos]/=(1+sigma);
				yc[pos]/=(1+sigma);*/
				//------------------------
				error+=SQR(yr[pos]-prevGrad[0])+SQR(yc[pos]-prevGrad[1]);
			}
		}
		error=sqrt(error/(nrows*ncols));
		cerr<<error<<endl;
		//update primal field
		computeDivergence(yr, yc, nrows, ncols, divergence, EDT_Backward, EBC_Circular);
		pos=0;
		
		for(int i=0;i<nrows;++i){
			for(int j=0;j<ncols;++j, ++pos){
				double diff=-x[pos];
				x[pos]+=tau*divergence[pos];
				if(pos==0){
					x[pos]=0;
				}
				diff+=x[pos];
				//update xbar
				xbar[pos]=x[pos]+theta*diff;
			}
		}
		
	}
	
	delete[] xbar;
	delete[] yr;
	delete[] yc;
	delete[] dxdr;
	delete[] dxdc;
	delete[] divergence;
}



double wrap(double x){
	int n=0;
	if(x<-M_PI){
		n=int(double(x-M_PI)/double(2*M_PI));
	}else if(x>M_PI){
		n=int(double(x+M_PI)/double(2*M_PI));
	}
	x-=2*n*M_PI;
	if(x<-M_PI){
		cerr<<x<<endl;
	}else if(x>M_PI){
		cerr<<x<<endl;
	}
	return x;
}

void wrap(double *x, int n, double *dest){
	for(int i=0;i<n;++i){
		dest[i]=wrap(x[i]);
	}
}


/*
	solves min_f lambda*|f-g|_1 +|Df|_1, where f and g are 2D vector fields and D is the differential oferator (Jacobian)
*/
void filterVectorField(double *gr, double *gc, int nrows, int ncols, double lambda, double tau, double sigma, double theta, double *fr, double *fc){
	double *frbar=new double[nrows*ncols];
	double *fcbar=new double[nrows*ncols];
	//---dual variable (Jacobian field)----
	double *yrr=new double[nrows*ncols];
	double *ycc=new double[nrows*ncols];
	double *yrc=new double[nrows*ncols];
	//---auxiliar Jacobian field (will hold the Jacobian of [frbar, fcbar]), assuming symmetry---
	double *dfrdr=new double[nrows*ncols];
	double *dfrdc=new double[nrows*ncols];
	double *dfcdc=new double[nrows*ncols];
	//---divergence of rank 2 tensor field---
	double *divr=new double[nrows*ncols];
	double *divc=new double[nrows*ncols];

	double tolerance=1e-8;
	double error=1+tolerance;
	int maxIter=5000;
	int iter=0;
	//---initialize dual variable---
	memset(yrr, 0, sizeof(double)*nrows*ncols);
	memset(ycc, 0, sizeof(double)*nrows*ncols);
	memset(yrc, 0, sizeof(double)*nrows*ncols);
	//---initialize intermediate iteration---
	memcpy(frbar, fr, sizeof(double)*nrows*ncols);
	memcpy(fcbar, fc, sizeof(double)*nrows*ncols);
	double errorPrev=1e10;
	double tau_lambda=tau*lambda;
	double tau_lambda2=tau_lambda*tau_lambda;
	while((tolerance<error) && (iter<=maxIter)){
		++iter;
		//---compute error---
		computeJacobian(fr, fc, nrows, ncols, dfrdr, dfcdc, dfrdc, EDT_Forward, EBC_Circular);
		error=0;
		int pos=0;
		for(int i=0;i<nrows;++i){
			for(int j=0;j<ncols;++j, ++pos){
				error+=sqrt(SQR(fr[pos]-gr[pos])+SQR(fc[pos]-gc[pos]));
			}
		}
		error*=lambda;
		pos=0;
		for(int i=0;i<nrows;++i){
			for(int j=0;j<ncols;++j, ++pos){
				error+=sqrt(SQR(dfrdr[pos]) + SQR(dfcdc[pos]) + 2*SQR(dfrdc[pos]));
			}
		}
		cerr<<error<<endl;

		//update dual field
		computeJacobian(frbar, fcbar, nrows, ncols, dfrdr, dfcdc, dfrdc, EDT_Forward, EBC_Circular);
		pos=0;
		for(int i=0;i<nrows;++i){
			for(int j=0;j<ncols;++j, ++pos){
				yrr[pos]+=sigma*dfrdr[pos];
				ycc[pos]+=sigma*dfcdc[pos];
				yrc[pos]+=sigma*dfrdc[pos];
				//----L1-L2---------------
				double nrm=yrr[pos]*yrr[pos] + ycc[pos]*ycc[pos]+2*yrc[pos]*yrc[pos];
				if(nrm>1){
					nrm=sqrt(nrm);
					yrr[pos]/=nrm;
					ycc[pos]/=nrm;
					yrc[pos]/=nrm;
				}
				//------------------------
			}
		}
		//update primal field
		computeDivergence(yrr,ycc,yrc,nrows, ncols, divr, divc, EDT_Backward, EBC_Circular);
		pos=0;
		//error=0;
		for(int i=0;i<nrows;++i){
			for(int j=0;j<ncols;++j, ++pos){
				double diffr=-fr[pos];
				double diffc=-fc[pos];
				fr[pos]+=tau*divr[pos];
				fc[pos]+=tau*divc[pos];
				//---compute proximal operator on fr[pos], fc[pos]---
				double nrm=SQR(fr[pos]-gr[pos])+SQR(fc[pos]-gc[pos]);
				if(nrm<=tau_lambda2){
					fr[pos]=gr[pos];
					fc[pos]=gc[pos];
				}else{
					nrm=sqrt(nrm);
					fr[pos]+=tau_lambda*(gr[pos]-fr[pos])/nrm;
					fc[pos]+=tau_lambda*(gc[pos]-fc[pos])/nrm;
				}
				//---------------------------------------------------
				diffr+=fr[pos];
				diffc+=fc[pos];
				//error+=SQR(diffr)+SQR(diffc);
				//update xbar
				frbar[pos]=fr[pos]+theta*diffr;
				fcbar[pos]=fc[pos]+theta*diffc;
			}
		}
		//error/=(nrows*ncols);
		//cerr<<error<<endl;
	}
	
	delete[] frbar;
	delete[] fcbar;
	delete[] yrr;
	delete[] ycc;
	delete[] yrc;
	delete[] dfrdr;
	delete[] dfrdc;
	delete[] dfcdc;
	delete[] divr;
	delete[] divc;
}


void filterVectorField_orientationInvariant(double *gr, double *gc, int nrows, int ncols, double lambda, double tau, double sigma, double theta, double *fr, double *fc){
	lambda*=-1;
	double *frbar=new double[nrows*ncols];
	double *fcbar=new double[nrows*ncols];
	//---dual variable (Jacobian field)----
	double *yrr=new double[nrows*ncols];
	double *ycc=new double[nrows*ncols];
	double *yrc=new double[nrows*ncols];
	//---auxiliar Jacobian field (will hold the Jacobian of [frbar, fcbar]), assuming symmetry---
	double *dfrdr=new double[nrows*ncols];
	double *dfrdc=new double[nrows*ncols];
	double *dfcdc=new double[nrows*ncols];
	//---divergence of rank 2 tensor field---
	double *divr=new double[nrows*ncols];
	double *divc=new double[nrows*ncols];
	//---precomputed norm of vector field g---
	double *normg=new double[nrows*ncols];
	int pos=0;
	for(int i=0;i<nrows;++i){
		for(int j=0;j<ncols;++j, ++pos){
			normg[pos]=gr[pos]*gr[pos] + gc[pos]*gc[pos];
			if(normg[pos]<1e-9){
				fr[pos]=0;
				fc[pos]=0;
			}else{
				fr[pos]=gr[pos]/sqrt(normg[pos]+1e-9);
				fc[pos]=gc[pos]/sqrt(normg[pos]+1e-9);
			}
			
		}
	}

	double tolerance=1e-8;
	double error=1+tolerance;
	int maxIter=50000;
	int iter=0;
	//---initialize dual variable---
	memset(yrr, 0, sizeof(double)*nrows*ncols);
	memset(ycc, 0, sizeof(double)*nrows*ncols);
	memset(yrc, 0, sizeof(double)*nrows*ncols);
	//---initialize intermediate iteration---
	memcpy(frbar, fr, sizeof(double)*nrows*ncols);
	memcpy(fcbar, fc, sizeof(double)*nrows*ncols);
	double errorPrev=1e10;
	double tau_lambda=tau*lambda;
	double tau_lambda2=tau_lambda*tau_lambda;
	while(/*(tolerance<error) &&*/ (iter<=maxIter)){
		++iter;
		//---compute error---
		computeJacobian(fr, fc, nrows, ncols, dfrdr, dfcdc, dfrdc, EDT_Forward, EBC_Circular);
		error=0;
		int pos=0;
		for(int i=0;i<nrows;++i){
			for(int j=0;j<ncols;++j, ++pos){
				double nrm=sqrt(fr[pos]*fr[pos] + fc[pos]*fc[pos]);
				if(nrm>1){//this should never happen
					cerr<<"Warning: vector norm greater than one."<<endl;
				}
				error+=fabs(gr[pos]*fr[pos] + gc[pos]*fc[pos]);
			}
		}
		error*=lambda;
		pos=0;
		for(int i=0;i<nrows;++i){
			for(int j=0;j<ncols;++j, ++pos){
				error+=sqrt(SQR(dfrdr[pos]) + SQR(dfcdc[pos]) + 2*SQR(dfrdc[pos]));
			}
		}
		cerr<<error<<endl;

		//update dual field
		computeJacobian(frbar, fcbar, nrows, ncols, dfrdr, dfcdc, dfrdc, EDT_Forward, EBC_Circular);
		pos=0;
		for(int i=0;i<nrows;++i){
			for(int j=0;j<ncols;++j, ++pos){
				yrr[pos]+=sigma*dfrdr[pos];
				ycc[pos]+=sigma*dfcdc[pos];
				yrc[pos]+=sigma*dfrdc[pos];
				//----L1-L2---------------
				double nrm=yrr[pos]*yrr[pos] + ycc[pos]*ycc[pos]+2*yrc[pos]*yrc[pos];
				if(nrm>1){
					nrm=sqrt(nrm);
					yrr[pos]/=nrm;
					ycc[pos]/=nrm;
					yrc[pos]/=nrm;
				}
				//------------------------
			}
		}
		//update primal field
		computeDivergence(yrr,ycc,yrc,nrows, ncols, divr, divc, EDT_Backward, EBC_Circular);
		pos=0;
		//error=0;
		for(int i=0;i<nrows;++i){
			for(int j=0;j<ncols;++j, ++pos){
				double diffr=-fr[pos];
				double diffc=-fc[pos];
				fr[pos]+=tau*divr[pos];
				fc[pos]+=tau*divc[pos];
				//---compute proximal operator on fr[pos], fc[pos]---
				if(normg[pos]<1e-9){//almost zero
					double nrm=fr[pos]*fr[pos] + fc[pos]*fc[pos];
					if(nrm>1){
						nrm=sqrt(nrm);
						fr[pos]/=nrm;
						fc[pos]/=nrm;
					}
				}else{
					double alpha=(gr[pos]*fr[pos]+gc[pos]*fc[pos])/(tau*lambda*normg[pos]);
					if(alpha>1){
						fr[pos]-=tau*lambda*gr[pos];
						fc[pos]-=tau*lambda*gc[pos];
					}else if(alpha<-1){
						fr[pos]+=tau*lambda*gr[pos];
						fc[pos]+=tau*lambda*gc[pos];
					}else{//|alpha|<=1
						fr[pos]-=tau*lambda*alpha*gr[pos];
						fc[pos]-=tau*lambda*alpha*gc[pos];
					}
					double beta=fr[pos]*fr[pos] + fc[pos]*fc[pos];
					if(beta>=1){
						beta=sqrt(beta);
						fr[pos]/=beta;
						fc[pos]/=beta;
					}
				}
				//---------------------------------------------------
				diffr+=fr[pos];
				diffc+=fc[pos];
				//error+=SQR(diffr)+SQR(diffc);
				//update xbar
				frbar[pos]=fr[pos]+theta*diffr;
				fcbar[pos]=fc[pos]+theta*diffc;
			}
		}
		//error/=(nrows*ncols);
		//cerr<<error<<endl;
	}
	
	delete[] frbar;
	delete[] fcbar;
	delete[] yrr;
	delete[] ycc;
	delete[] yrc;
	delete[] dfrdr;
	delete[] dfrdc;
	delete[] dfcdc;
	delete[] divr;
	delete[] divc;
	delete[] normg;
}



/*
	solves min_f lambda*|f-g|_1 +|Df|_1, where f and g are 3D vector fields and D is the differential oferator (Jacobian)
*/
void filterVectorField(double *gs, double *gr, double *gc, int nslices, int nrows, int ncols, double lambda, double tau, double sigma, double theta, double *fs, double *fr, double *fc){
	int nvox=nslices*nrows*ncols;
	double *fsbar=new double[nvox];
	double *frbar=new double[nvox];
	double *fcbar=new double[nvox];
	//---dual variable (Jacobian field)----
	double *yss=new double[nvox];
	double *yrr=new double[nvox];
	double *ycc=new double[nvox];
	double *ysr=new double[nvox];
	double *yrc=new double[nvox];
	double *ysc=new double[nvox];
	//---auxiliar Jacobian field (will hold the Jacobian of [frbar, fcbar]), assuming symmetry---
	double *dfsds=new double[nvox];
	double *dfrdr=new double[nvox];
	double *dfcdc=new double[nvox];
	double *dfsdr=new double[nvox];
	double *dfrdc=new double[nvox];
	double *dfsdc=new double[nvox];
	//---divergence of rank 2 and dimension 3 tensor field---
	double *divs=new double[nvox];
	double *divr=new double[nvox];
	double *divc=new double[nvox];

	double tolerance=1e-8;
	double error=1+tolerance;
	int maxIter=5000;
	int iter=0;
	//---initialize dual variable---
	memset(yss, 0, sizeof(double)*nvox);
	memset(yrr, 0, sizeof(double)*nvox);
	memset(ycc, 0, sizeof(double)*nvox);
	memset(ysr, 0, sizeof(double)*nvox);
	memset(yrc, 0, sizeof(double)*nvox);
	memset(ysc, 0, sizeof(double)*nvox);
	//---initialize intermediate iteration---
	memcpy(fsbar, fs, sizeof(double)*nvox);
	memcpy(frbar, fr, sizeof(double)*nvox);
	memcpy(fcbar, fc, sizeof(double)*nvox);
	double errorPrev=1e10;
	double tau_lambda=tau*lambda;
	double tau_lambda2=tau_lambda*tau_lambda;
	while((tolerance<error) && (iter<=maxIter)){
		++iter;
		//---compute error---
		computeJacobian(fs, fr, fc, nslices, nrows, ncols, dfsds, dfrdr, dfcdc, dfsdr, dfrdc, dfsdc, EDT_Forward, EBC_Circular);
		error=0;
		for(int pos=0;pos<nvox;++pos){
			error+=sqrt(SQR(fr[pos]-gr[pos])+SQR(fc[pos]-gc[pos]));
		}
		error*=lambda;
		for(int pos=0;pos<nvox;++pos){
			error+=sqrt(SQR(dfrdr[pos]) + SQR(dfcdc[pos]) + 2*SQR(dfrdc[pos]));
		}
		cerr<<iter<<"/"<<maxIter<<": "<<error<<endl;
		//update dual field
		computeJacobian(fsbar, frbar, fcbar, nslices, nrows, ncols, dfsds, dfrdr, dfcdc, dfsdr, dfrdc, dfsdc, EDT_Forward, EBC_Circular);
		int pos=0;
		for(int s=0;s<nslices;++s){
			for(int i=0;i<nrows;++i){
				for(int j=0;j<ncols;++j, ++pos){
					yss[pos]+=sigma*dfsds[pos];
					yrr[pos]+=sigma*dfrdr[pos];
					ycc[pos]+=sigma*dfcdc[pos];
					ysr[pos]+=sigma*dfsdr[pos];
					yrc[pos]+=sigma*dfrdc[pos];
					ysc[pos]+=sigma*dfsdc[pos];
					//----L1-L2---------------
					double nrm=		yss[pos]*yss[pos] + yrr[pos]*yrr[pos] + ycc[pos]*ycc[pos]+
								2*(	ysr[pos]*ysr[pos] + yrc[pos]*yrc[pos] + ysc[pos]*ysc[pos]);
					if(nrm>1){
						nrm=sqrt(nrm);
						yss[pos]/=nrm;
						yrr[pos]/=nrm;
						ycc[pos]/=nrm;
						ysr[pos]/=nrm;
						yrc[pos]/=nrm;
						ysc[pos]/=nrm;
					}
					//------------------------
				}
			}
		}
		//update primal field
		computeDivergence(ycc, yrr, ycc, ysr, yrc, ysc, nslices, nrows, ncols, divs, divr, divc, EDT_Backward, EBC_Circular);
		pos=0;
		for(int s=0;s<nslices;++s){
			for(int i=0;i<nrows;++i){
				for(int j=0;j<ncols;++j, ++pos){
					double diffs=-fs[pos];
					double diffr=-fr[pos];
					double diffc=-fc[pos];
					fs[pos]+=tau*divs[pos];
					fr[pos]+=tau*divr[pos];
					fc[pos]+=tau*divc[pos];
					//---compute proximal operator on fs[pos], fr[pos], fc[pos]---
					double nrm=SQR(fs[pos]-gs[pos]) + SQR(fr[pos]-gr[pos]) + SQR(fc[pos]-gc[pos]);
					if(nrm<=tau_lambda2){
						fs[pos]=gs[pos];
						fr[pos]=gr[pos];
						fc[pos]=gc[pos];
					}else{
						nrm=sqrt(nrm);
						fs[pos]+=tau_lambda*(gs[pos]-fs[pos])/nrm;
						fr[pos]+=tau_lambda*(gr[pos]-fr[pos])/nrm;
						fc[pos]+=tau_lambda*(gc[pos]-fc[pos])/nrm;
					}
					//---------------------------------------------------
					diffs+=fs[pos];
					diffr+=fr[pos];
					diffc+=fc[pos];
					//update xbar
					fsbar[pos]=fs[pos]+theta*diffs;
					frbar[pos]=fr[pos]+theta*diffr;
					fcbar[pos]=fc[pos]+theta*diffc;
				}
			}
		}
	}
	delete[] fsbar;
	delete[] frbar;
	delete[] fcbar;
	delete[] yss;
	delete[] yrr;
	delete[] ycc;
	delete[] ysr;
	delete[] yrc;
	delete[] ysc;
	delete[] dfsds;
	delete[] dfrdr;
	delete[] dfcdc;
	delete[] dfsdr;
	delete[] dfrdc;
	delete[] dfsdc;
	delete[] divs;
	delete[] divr;
	delete[] divc;
}


void filterMultiVectorField2D(double *gr, double *gc, int *assignmentV, int *assignmentH, int nrows, int ncols, int maxCompartments, double lambda, double tau, double sigma, double theta, double *fr, double *fc){
	double *frbar=new double[nrows*ncols*maxCompartments];
	double *fcbar=new double[nrows*ncols*maxCompartments];
	//---dual variable (Jacobian field)----
	double *yrr=new double[nrows*ncols*maxCompartments];
	double *ycc=new double[nrows*ncols*maxCompartments];
	double *yrc=new double[nrows*ncols*maxCompartments];
	//---auxiliar Jacobian field (will hold the Jacobian of [frbar, fcbar]), assuming symmetry---
	double *dfrdr=new double[nrows*ncols*maxCompartments];
	double *dfrdc=new double[nrows*ncols*maxCompartments];
	double *dfcdc=new double[nrows*ncols*maxCompartments];
	//---divergence of rank 2 tensor field---
	double *divr=new double[nrows*ncols*maxCompartments];
	double *divc=new double[nrows*ncols*maxCompartments];

	double tolerance=1e-8;
	double error=1+tolerance;
	int maxIter=5000;
	int iter=0;
	//---initialize dual variable---
	memset(yrr, 0, sizeof(double)*nrows*ncols*maxCompartments);
	memset(ycc, 0, sizeof(double)*nrows*ncols*maxCompartments);
	memset(yrc, 0, sizeof(double)*nrows*ncols*maxCompartments);
	//---initialize intermediate iteration---
	memcpy(frbar, fr, sizeof(double)*nrows*ncols*maxCompartments);
	memcpy(fcbar, fc, sizeof(double)*nrows*ncols*maxCompartments);
	double errorPrev=1e10;
	double tau_lambda=tau*lambda;
	double tau_lambda2=tau_lambda*tau_lambda;
	while((tolerance<error) && (iter<=maxIter)){
		++iter;
		//---compute error---
		computeMultiJacobian(fr, fc, assignmentV, assignmentH, nrows, ncols, maxCompartments, dfrdr, dfcdc, dfrdc, EDT_Forward, EBC_Circular);
		error=0;
		int pos=0;
		for(int i=0;i<nrows;++i){
			for(int j=0;j<ncols;++j){
				for(int k=0;k<maxCompartments;++k, ++pos){
					error+=sqrt(SQR(fr[pos]-gr[pos])+SQR(fc[pos]-gc[pos]));
					if(!isNumber(error)){
						cerr<<i<<", "<<j<<", "<<k<<endl;
					}
				}
			}
		}
		error*=lambda;
		pos=0;
		for(int i=0;i<nrows;++i){
			for(int j=0;j<ncols;++j){
				for(int k=0;k<maxCompartments;++k, ++pos){
					error+=sqrt(SQR(dfrdr[pos]) + SQR(dfcdc[pos]) + 2*SQR(dfrdc[pos]));
				}
			}
		}
		cerr<<error<<endl;

		//update dual field
		computeMultiJacobian(frbar, fcbar, assignmentV, assignmentH, nrows, ncols, maxCompartments, dfrdr, dfcdc, dfrdc, EDT_Forward, EBC_Circular);
		pos=0;
		for(int i=0;i<nrows;++i){
			for(int j=0;j<ncols;++j){
				for(int k=0;k<maxCompartments;++k, ++pos){
					yrr[pos]+=sigma*dfrdr[pos];
					ycc[pos]+=sigma*dfcdc[pos];
					yrc[pos]+=sigma*dfrdc[pos];
					//----L1-L2---------------
					double nrm=yrr[pos]*yrr[pos] + ycc[pos]*ycc[pos]+2*yrc[pos]*yrc[pos];//<<<<
					if(nrm>1){
						nrm=sqrt(nrm);
						yrr[pos]/=nrm;
						ycc[pos]/=nrm;
						yrc[pos]/=nrm;
					}
					//------------------------
				}
			}
		}
		//update primal field
		computeMultiDivergence(yrr, ycc, yrc, assignmentV, assignmentH, nrows, ncols, maxCompartments, divr, divc, EDT_Backward, EBC_Circular);
		pos=0;
		//error=0;
		for(int i=0;i<nrows;++i){
			for(int j=0;j<ncols;++j){
				for(int k=0;k<maxCompartments;++k, ++pos){
					double diffr=-fr[pos];
					double diffc=-fc[pos];
					fr[pos]+=tau*divr[pos];
					fc[pos]+=tau*divc[pos];
					//---compute proximal operator on fr[pos], fc[pos]---
					double nrm=SQR(fr[pos]-gr[pos])+SQR(fc[pos]-gc[pos]);
					if(nrm<=tau_lambda2){
						fr[pos]=gr[pos];
						fc[pos]=gc[pos];
					}else{
						nrm=sqrt(nrm);
						fr[pos]+=tau_lambda*(gr[pos]-fr[pos])/nrm;
						fc[pos]+=tau_lambda*(gc[pos]-fc[pos])/nrm;
					}
					//---------------------------------------------------
					diffr+=fr[pos];
					diffc+=fc[pos];
					//error+=SQR(diffr)+SQR(diffc);
					//update xbar
					frbar[pos]=fr[pos]+theta*diffr;
					fcbar[pos]=fc[pos]+theta*diffc;
				}
				
			}
		}
		//error/=(nrows*ncols);
		//cerr<<error<<endl;
	}
	
	delete[] frbar;
	delete[] fcbar;
	delete[] yrr;
	delete[] ycc;
	delete[] yrc;
	delete[] dfrdr;
	delete[] dfrdc;
	delete[] dfcdc;
	delete[] divr;
	delete[] divc;
}

