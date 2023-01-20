//
//  main.cpp
//  E_I_model_with_synaptic_current_2D
//  Created by KANGLING on 2021/11/15.
//

 
#include<iostream>
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>
#include<random>
#include<algorithm>
std::random_device rd; int seed = rd();
std::mt19937 rng(seed);
std::uniform_real_distribution<double> real_random(-0.1,0.1);
std::normal_distribution<double> normal_distribution(0,1.0);

/*
 Two-dimensional network structure
 t_tau: distance-dependent propagation delay
 weight: long-range excitatory connection strength
 color: different kinds of modules
 */
typedef struct nw
{
    int **t_tau;
    double **weight;
    int *color;
} snw;
/*
 Rate model f-I curve and adaptive time scale
 N_data: length of the data
 origin_data: [0] current; [1] firing rate; [2]: time scale
 */
typedef struct f_tau
{
    int N_data;
    double **origin_data;
    
} sf_tau;
/*Functions*/
/* Get current, firing rate and time scale from original data list*/
sf_tau *read_data(double i_min, double i_max,double d_i);
double function_chose_frequency(int N,double current,double**origin_data);
double function_chose_tau(int N,double current,double**origin_data);
double function_chose_current(int N,double r,double**origin_data);
/*Network*/
snw*network_2D(int N,double lambda,int tau );
/*External current*/
double function_i_e_ext(double i_e_ext0, double sigma_ext_e, double eta_ind);
double function_i_i_ext(double i_i_ext0, double sigma_ext_i, double eta_ind);
double function_eta(double eta,double tau_ext,double xi_ind,double xi_all,double eta_c );
/* The kinetic kernels the synaptic current*/
double function_current_arise(int i, int NN, double current_arise,double tau_arise, double r);
double function_current_decay(int i, int NN, double current_decay, double tau_decay,double current_arise);
/*E-I modules*/
double function_i_e(int i,int NN,double i_e, double tau_e, double i_e_ext, double omega_ee,double omega_ei, double *current_decay_e, double current_decay_i, double**weight);
double function_i_i(int i,int NN,double i_i, double tau_i, double i_i_ext, double omega_ie,double omega_ii, double *current_decay_e, double current_decay_i, double **weight);
double delay_function_i_e(int i,int NN,double i_e, double tau_e, double i_e_ext, double omega_ee,double omega_ei, double *current_decay_e, double current_decay_i,double**weight,double**current_decay_e_tem,int **t_tau);
double delay_function_i_i(int i,int NN,double i_i, double tau_i, double i_i_ext, double omega_ie,double omega_ii, double *current_decay_e, double current_decay_i,double **weight,double **current_decay_e_tem,int **t_tau);
/*Simulation*/
double calculate(int max_step,int N_data,double**origin_data,int NN,double **weight,double lambda,double i_e_ext0, double i_i_ext0,double tau_ext,double sigma_ext_e,double sigma_ext_i,double omega_ee, double omega_ei,double omega_ie,double omega_ii,int N_noise,int **t_tau,int max_t_tau,double tau,double eta_c, int *color );

int main()
{
    
    /*Recording simulation time*/
    clock_t startTime,endTime;
    startTime = clock();
    
    /* Generating network
     N (NN): network size
     tau: basic propagation delay (unit: ms/step)
     t_tau: distance-dependent propagation delay
     lambda: excitatory connectivity range
     weight: long-range excitatory connection strength
     color: different kinds of modules
     */
    snw*snwk;
    double **weight;
    int N=10;
    int NN=N*N;
    int **t_tau;
    int *color;
    double tau=130;
    double lambda=2.0;
    
    snwk=network_2D(N,lambda,tau );
    weight=(snwk->weight);
    t_tau=(snwk->t_tau);
    color=(snwk->color);

    FILE*fp_weight;
    char filename_weight[256];
    sprintf(filename_weight,"%d_%.2f_weight.txt",N,lambda);
    fp_weight=fopen(filename_weight,"w");
    for(int i=0;i<NN;i++)
    {
        double add_weight=0.0;
        for(int j=0;j<NN;j++)
        {
            add_weight+=weight[i][j];
            fprintf(fp_weight,"%f,",weight[i][j]);
        }
        fprintf(fp_weight,"%f,\n",add_weight);
    }
    int max_t_tau=0;
    for(int i=0;i<NN;i++)
    {
        for(int j=0;j<NN;j++)
        {
            if(t_tau[i][j]>max_t_tau) max_t_tau=t_tau[i][j];
            fprintf(fp_weight,"%d,",t_tau[i][j]);
        }
        fprintf(fp_weight,"\n");
    }
    fclose(fp_weight);
    printf( "max_t_tau=%d\n",max_t_tau);
    
    FILE * fp_color;
    char filename_color[256];
    sprintf(filename_color, "%d_%f_color.txt", N,lambda);
    fp_color = fopen(filename_color, "w");
    for (int i = 0; i < N; i++)
    {
        
        for (int j = 0; j <N; j++)
        {
            int jj= i*N+j;
            fprintf(fp_color, "%d\t", color[jj]);
        }
        fprintf(fp_color, "\n");
    }
    fclose(fp_color);
    
    /*  Reading original data of rate model, f-I curve and adaptive time scale*/
    double i_min=-20.0;
    double i_max=100.0;
    double d_i=0.1;
    sf_tau*sf_tauk;
    double**origin_data;
    int N_data;
    sf_tauk = read_data(i_min,i_max,d_i);
    origin_data = (sf_tauk->origin_data);
    N_data = (sf_tauk->N_data);
    
    FILE*fp_f_tau;
    fp_f_tau = fopen("reorigin_data.txt", "w");
    for (int i = 0; i < N_data; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            
            fprintf(fp_f_tau,"%f,", origin_data[i][j]);
        }
        fprintf(fp_f_tau,"\n");
    }
    fclose(fp_f_tau);
    
    /*Finite-size noise*/
    int N_noise=0000;
    
    /*Recurrent synaptic coupling strength (mV.s)*/
    double omega_ee=0.96;
    double omega_ie=1.0;
    double omega_ei=2.08;
    double omega_ii=0.87;
    
    /*
     External input
     eta_c: proportion of global external inputs
     niu (Hz): external input amplitude 2uctuations
     tau_ext (ms): correlation time of external input fluctuations
     */
    double niu=0;
    double sigma_ext_e= niu*omega_ee;
    double sigma_ext_i= 2*niu*omega_ie;
    double eta_c=1.0;
    double tau_ext=25;
    
 
    /*Steady state*/
    double r_e_s=5.0;
    double r_i_s=10.0;
    double i_e_s=0 ;
    double i_i_s=0 ;
    i_e_s=function_chose_current(N_data, r_e_s,origin_data);
    i_i_s=function_chose_current(N_data, r_i_s,origin_data);
    double i_e_ext0=i_e_s-omega_ee*r_e_s+omega_ei*r_i_s;
    double i_i_ext0=i_i_s-omega_ie*r_e_s+omega_ii*r_i_s;
    // printf("%f,%f,%f,%f,%f,%f\n",r_e_s,r_i_s,i_e_s,i_i_s, i_e_ext0,i_i_ext0);
    
    /*Simulation*/
    int max_step=500*100;
    calculate(max_step, N_data, origin_data, NN, weight,lambda,i_e_ext0, i_i_ext0, tau_ext,sigma_ext_e,sigma_ext_i, omega_ee, omega_ei, omega_ie, omega_ii, N_noise,t_tau,max_t_tau,tau,eta_c ,color);
    
    endTime = clock();
    std::cout << "Totle Time : " <<(double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << std::endl;
    
    // printf("end");
    // getchar();
    // return 0;
}

snw*network_2D(int N,double lambda,int tau )
{
    int NN=N*N;
    snw*snwk;
    snwk=(snw*)malloc(1*sizeof(snw));
    int **degree;
    int **adjacent;
    double **weight;
    int **t_tau;
    
    t_tau=(int**)malloc(NN*sizeof(int*));
    degree=(int**)malloc(NN*sizeof(int*));
    adjacent=(int**)malloc(NN*sizeof(int*));
    weight=(double**)malloc(NN*sizeof(double*));
    for (int i = 0; i < NN; i++)
    {
        adjacent[i] = (int*)malloc(NN * sizeof(int));
        degree[i] = (int*)malloc(NN * sizeof(int));
        weight[i] = (double*)malloc(NN * sizeof(double));
        t_tau[i] = (int*)malloc(NN*sizeof(int));
        
        for(int j=0;j<NN;j++)
        {
            weight[i][j]=0.0;
            adjacent[i][j] = 0;
            degree[i][j] = 0;
            t_tau[i][j]=0;
        }
    }
    for (int i = 0; i < NN; i++)
    {
        adjacent[i][i]=1;
        degree[i][0]++;
        degree[i][degree[i][0]] = i;
        for (int j = i+1; j < NN; j++)
        {
            adjacent[i][j] = adjacent[j][i] = 1;
            degree[i][0]++;
            degree[i][degree[i][0]] = j;
            degree[j][0]++;
            degree[j][degree[j][0]] = i;
        }
    }
    
    int *row;
    int *col;
    row = (int*)malloc(NN * sizeof(int));
    col = (int*)malloc(NN * sizeof(int));
    for (int i = 0; i < NN; i++)
    {
        row[i] = 0;
        col[i] = 0;
    }
    for (int i = 0; i < NN; i++)
    {
        row[i] = int(i / N);
        col[i] = i % N;
    }
    
    double **distance;
    distance = (double**)malloc(NN * sizeof(double*));
    for (int i = 0; i < NN; i++)
    {
        distance[i] = (double*)malloc(NN * sizeof(double));
        for (int j = 0; j < NN; j++)
        {
            distance[i][j] = 0;
        }
    }
    int   row_d;
    int   col_d;
    for (int i = 0; i < NN; i++)
    {
        for (int j = 0; j <NN; j++)
        {
            if (abs(row[j] - row[i]) > N/2)
            {
                if (row[j] - row[i] > 0)
                {
                    row_d = N - row[j] + row[i];
                }
                else
                {
                    row_d = N - row[i] + row[j];
                }
            }
            else
            {
                row_d = row[j] - row[i];
            }
            if (abs(col[j] - col[i]) > N/2)
            {
                if (col[j] - col[i] > 0)
                {
                    col_d = N - col[j] + col[i];
                }
                else
                {
                    col_d = N - col[i] + col[j];
                }
            }
            else
            {
                col_d = col[j] - col[i];
            }
            distance[i][j] = sqrt(row_d * row_d + col_d * col_d);
            // printf("%d,%d,%f\n", jj, i, distance[i][j]);
        }
    }
    
    double add_weight=0;
    for(int i=0;i<NN;i++)
    {
        for(int j=0;j<NN;j++)
        {
            double d=distance[i][j];
            weight[i][j]=weight[j][i]= exp(-((d/lambda)*(d/lambda)));
        }
        if(i==0)
        {
            for(int j=0;j<NN;j++)
            {
                add_weight+=weight[i][j];
            }
        }
    }

    for(int i=0;i<NN;i++)
    {
        for(int j=0;j<NN;j++)
        {
            weight[i][j]= weight[i][j]/add_weight;
        }
    }
    
    /*Finding the effective length of coupling*/
    int R=N;
    for(int i=0;i<1;i++)
    {
        for(int j=0;j<N;j++)
        {
            if (weight[i][j]<0.0000001)
            {
                R=j;
                break;
            }
        }
    }
    for(int i=0;i<NN;i++)
    {
        for(int j=0;j<NN;j++)
        {
            if (distance[i][j]<R)
            {
                t_tau[i][j]=t_tau[j][i]=(int)(tau*distance[i][j]);
            }
        }
    }
    FILE * fp_net;
    char filename_net[256];
    sprintf(filename_net, "%d_%.2f_distance.txt", N,lambda);
    fp_net = fopen(filename_net, "w");
    for (int i = 0; i < NN; i++)
    {
        for (int j = 0; j <NN; j++)
        {
            fprintf(fp_net, "%d,%d,%f,%f,%d\n", i, j, distance[i][j]*distance[i][j],weight[i][j],t_tau[i][j]);
        }
    }
    fclose(fp_net);
    
    /*Distinguishing different kinds modules*/
    int sur_width=0;
    int *color;
    color=(int*)malloc(NN*sizeof(int));
    for (int i = 0; i < NN; i++)
    {
        color[i] =0;
    }
    
    for (int i = 0; i < sur_width; i++)
    {
        for(int j=i*N; j<((i+1)*N);j++)
        {
            color[j]=1;
        }
        for(int j=0; j<N;j++)
        {
            int jj=j*N+i;
            color[jj]=1;
        }
    }
    
    for (int i = N-sur_width; i < N; i++)
    {
        for(int j=i*N; j<((i+1)*N);j++)
        {
            color[j]=1;
        }
        for(int j=0; j<N;j++)
        {
            int  jj=j*N+i;
            color[jj]=1;
        }
    }
    
    (snwk->color) = color;
    (snwk->t_tau) = t_tau;
    (snwk->weight)=weight;
    
    return snwk;
}

sf_tau *read_data(double i_min,double i_max,double di)
{
    int N=(int)((i_max-i_min)/di)+1;
    int n_bin=5; //reorder the bin;
    int re_N=N*n_bin-n_bin+1;
    sf_tau*sf_tauk;
    sf_tauk = (sf_tau*)malloc(sizeof(sf_tau));
    
    double**origin_data;
    double**reorigin_data;
    origin_data = (double**)malloc(N * sizeof(double*));
    reorigin_data = (double**)malloc(re_N * sizeof(double*));
    for (int i = 0; i < N; i++)
    {
        origin_data[i] = (double*)malloc(3 * sizeof(double));
        for (int j = 0; j < 3; j++)
        {
            origin_data[i][j] = 0.0;
        }
    }
    for (int i = 0; i < re_N; i++)
    {
        reorigin_data[i] = (double*)malloc(3 * sizeof(double));
        for (int j = 0; j < 3; j++)
        {
            reorigin_data[i][j] = 0.0;
        }
    }
    //**read the origin rates and tau
    FILE *fp;
    fp = fopen("origin_data.txt", "r");
    if (!fp)
    {
        printf("can't open file\n");
        getchar();
        exit;
    }
     
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            fscanf(fp, "%lf,", &origin_data[i][j]);
            // printf("%d,%f\n,", i,origin_data[i][j]);
        }
        origin_data[i][1]=origin_data[i][1]*1000;
    }
    fclose(fp);
    
    
    for (int i = 0; i < N; i++)
    {
        if(i<N-1)
        {
            for (int bin=0;bin<n_bin;bin++)
            {
                int re_i=i*n_bin+bin;
                if(bin==0)
                {
                    for (int j = 0; j < 3; j++)
                    {
                        reorigin_data[re_i][j]=origin_data[i][j];
                    }
                }
                else
                {
                    for (int j = 0; j < 3; j++)
                    {
                        reorigin_data[re_i][j]=(origin_data[i+1][j]-origin_data[i][j])*bin/n_bin+origin_data[i][j];
                    }
                }
            }
        }
    }
    reorigin_data[re_N-1][0]=origin_data[N-1][0];
    reorigin_data[re_N-1][1]=origin_data[N-1][1];
    reorigin_data[re_N-1][2]=origin_data[N-1][2];
   
    (sf_tauk->origin_data) = reorigin_data;
    (sf_tauk->N_data) = re_N;
    return sf_tauk;
}

double function_chose_tau(int N,double current,double**origin_data)
{
    double dis_current=origin_data[1][0]-origin_data[0][0];
    double div_bin=1000;
    double unit_div_bin=dis_current/div_bin;
    int dis=0;
    
    double dis_tau=0.0;
    double tau=0.0;
    if(current<-20.0)
    {
        tau=origin_data[0][2];
    }
    else if(current>100)
    {
        tau=origin_data[N-1][2];
    }
    else{
        
        for(int i=0;i<N;i++)
        {
            if((origin_data[i][0]-current)>0.0000)
            {
                dis=(int)((current-origin_data[i-1][0])/unit_div_bin);
                dis_tau=origin_data[i][2]-origin_data[i-1][2];
                tau=origin_data[i-1][2]+dis_tau/div_bin*dis;
                break;
            }
        }
    }
    return tau;
    
}

double function_chose_frequency(int N,double current,double**origin_data)
{
    double dis_current=origin_data[1][0]-origin_data[0][0];
    double div_bin=1000;
    double unit_div_bin=dis_current/div_bin;
    
    int dis=0;
    double dis_r=0.0;
    double r=0;
    if(current<-20.0)
    {
        r=origin_data[0][1];
    }
    else if(current>100)
    {
        r=origin_data[N-1][1];
    }
    else{
        
        for(int i=0;i<N;i++)
        {
            if((origin_data[i][0]-current)>0.0000)
            {
                dis=(int)((current-origin_data[i-1][0])/unit_div_bin);
                dis_r=origin_data[i][1]-origin_data[i-1][1];
                r=origin_data[i-1][1]+dis_r/div_bin*dis;
                break;
            }
        }
    }
    return r;
    
}

double function_chose_current(int N,double r,double**origin_data)
{
    double dis_r=origin_data[1][1]-origin_data[0][1];
    double div_bin=1000;
    double unit_div_bin=dis_r/div_bin;
     
    int dis=0;
    double dis_current=0.0;
    double current=0;
    if(r< 0)
    {
        r=origin_data[0][0];
    }
    else if(r>262.2737)
    {
        r=origin_data[N-1][0];
    }
    else{
        
        for(int i=0;i<N;i++)
        {
            
            if((origin_data[i][1]-r)>0.0000)
            {
                dis=(int)((r-origin_data[i-1][1])/unit_div_bin);
                dis_current=origin_data[i][0]-origin_data[i-1][0];
                current=origin_data[i-1][0]+dis_r/div_bin*dis;
                break;
            }
        }
    }
    return current;
}

double function_i_e_ext(double i_e_ext0, double sigma_ext_e, double eta_ind )
{
    double f_i_e_ext=0;
    f_i_e_ext= i_e_ext0+sigma_ext_e*eta_ind;
    return f_i_e_ext;
}

double function_i_i_ext(double i_i_ext0, double sigma_ext_i,double eta_ind)
{
    double f_i_i_ext=0;
    f_i_i_ext= i_i_ext0+sigma_ext_i*eta_ind;
    return f_i_i_ext;
}

double function_eta(double eta,double tau_ext,double xi_ind,double xi_all,double eta_c )
{
    double f_eta=0;
    f_eta=1.0/tau_ext*(-eta+sqrt(tau_ext)*(sqrt(1.0-eta_c)*xi_ind+sqrt(eta_c)*xi_all)) ;
    return f_eta;
}

double function_current_arise(  double current_arise,double tau_arise, double r)
{
    double f_current_arise;
    f_current_arise=1.0/tau_arise*(-current_arise+r);
    return f_current_arise;
}

double function_current_decay(  double current_decay, double tau_decay,double current_arise)
{
    double f_current_decay;
    f_current_decay=1.0/tau_decay*(-current_decay+current_arise );
    return f_current_decay;
}

double function_i_e(int i,int NN,double i_e, double tau_e, double i_e_ext, double omega_ee,double omega_ei, double *current_decay_e, double current_decay_i,double**weight)
{
    double f_i_e=0;
    double sum=0.0;
    for (int j=0;j<NN;j++)
    {
        sum+=weight[i][j]*current_decay_e[j];
    }
    f_i_e=1.0/tau_e*(-i_e+i_e_ext+omega_ee*sum-omega_ei*current_decay_i);
    return f_i_e;
}

double function_i_i(int i,int NN,double i_i, double tau_i, double i_i_ext, double omega_ie,double omega_ii, double *current_decay_e, double current_decay_i, double **weight)
{
    double f_i_i=0;
    double sum=0.0;
    for (int j=0;j<NN;j++)
    {
        sum+=weight[i][j]*current_decay_e[j];
    }
    f_i_i=1.0/tau_i*(-i_i+i_i_ext+omega_ie*sum-omega_ii*current_decay_i);
    return f_i_i;
}

double delay_function_i_e(int i,int NN,double i_e, double tau_e, double i_e_ext, double omega_ee,double omega_ei, double *current_decay_e, double current_decay_i,double**weight,double**current_decay_e_tem,int **t_tau)
{
    double f_i_e=0;
    double sum=0.0;
    for (int j=0;j<NN;j++)
    {
        sum+=weight[i][j]*current_decay_e_tem[j][t_tau[i][j]];
    }
    f_i_e=1.0/tau_e*(-i_e+i_e_ext+omega_ee*sum-omega_ei*current_decay_i);
    return f_i_e;
}

double delay_function_i_i(int i,int NN,double i_i, double tau_i, double i_i_ext, double omega_ie,double omega_ii, double *current_decay_e, double current_decay_i,double **weight,double **current_decay_e_tem,int **t_tau)
{
    double f_i_i=0;
    double sum=0.0;
    for (int j=0;j<NN;j++)
    {
        sum+=weight[i][j]*current_decay_e_tem[j][t_tau[i][j]];
    }
    f_i_i=1.0/tau_i*(-i_i+i_i_ext+omega_ie*sum-omega_ii*current_decay_i);
    return f_i_i;
}

double calculate(int max_step,int N_data,double**origin_data,int NN,double **weight,double lambda,double i_e_ext0, double i_i_ext0,double tau_ext,double sigma_ext_e,double sigma_ext_i,double omega_ee, double omega_ei,double omega_ie,double omega_ii,int N_noise,int **t_tau,int max_t_tau,double tau,double eta_c, int *color )
{
    
    FILE*fp;
    char filename[256];
    sprintf(filename,"%d_%.f_%d_%.2f_%.2f_%.2f_%.2f_%.2f_%.2f_%.2f_current_tau_fix.txt",NN,tau,N_noise,lambda,omega_ee,omega_ei,omega_ie,omega_ii,sigma_ext_e,eta_c );
    fp=fopen(filename,"w");
    
    double *i_e,*i_i,*r_e,*r_i,*tau_e,*tau_i;
    i_e=(double*)malloc(NN*sizeof(double));
    i_i=(double*)malloc(NN*sizeof(double));
    r_e=(double*)malloc(NN*sizeof(double));
    tau_e=(double*)malloc(NN*sizeof(double));
    r_i=(double*)malloc(NN*sizeof(double));
    tau_i=(double*)malloc(NN*sizeof(double));
    for(int i=0;i<NN;i++)
    {
        i_e[i]=-6.7;
        i_i[i]=0;
        r_e[i]=0.0;
        tau_e[i]=0.0;
        r_i[i]=0.0;
        tau_i[i]=0.0;
    }
    double *i_e_pre,*i_i_pre;
    i_e_pre=(double*)malloc(NN*sizeof(double));
    i_i_pre=(double*)malloc(NN*sizeof(double));
    
    for(int i=0;i<NN;i++)
    {
        i_e_pre[i]=0.0;
        i_i_pre[i]=0.0;
    }
    
    /*External current*/
    double *i_e_ext,*i_i_ext,*eta_all,*eta_ind,*eta_all_pre,*eta_ind_pre,*xi_all,*xi_ind;
    i_e_ext=(double*)malloc(NN*sizeof(double));
    i_i_ext=(double*)malloc(NN*sizeof(double));
    eta_all=(double*)malloc(NN*sizeof(double));
    eta_ind=(double*)malloc(NN*sizeof(double));
    eta_all_pre=(double*)malloc(NN*sizeof(double));
    eta_ind_pre=(double*)malloc(NN*sizeof(double));
    xi_all=(double*)malloc(NN*sizeof(double));
    xi_ind=(double*)malloc(NN*sizeof(double));
    for(int i=0;i<NN;i++)
    {
        i_e_ext[i]=0;
        i_i_ext[i]=0;
        eta_all[i]=real_random(rng);
        eta_ind[i]=real_random(rng);
        eta_all_pre[i]=0;
        eta_ind_pre[i]=0;
        xi_all[i]=0;
        xi_ind[i]=0;
    }
    
    /*Synatic current*/
    double *current_decay_e,*current_arise_e,*current_decay_i,*current_arise_i;
    double *current_decay_e_pre,*current_arise_e_pre,*current_decay_i_pre,*current_arise_i_pre;
    current_decay_e=(double*)malloc(NN*sizeof(double));
    current_arise_e=(double*)malloc(NN*sizeof(double));
    current_decay_i=(double*)malloc(NN*sizeof(double));
    current_arise_i=(double*)malloc(NN*sizeof(double));
    current_decay_e_pre=(double*)malloc(NN*sizeof(double));
    current_arise_e_pre=(double*)malloc(NN*sizeof(double));
    current_decay_i_pre=(double*)malloc(NN*sizeof(double));
    current_arise_i_pre=(double*)malloc(NN*sizeof(double));
    for(int i=0;i<NN;i++)
    {
        current_decay_e[i]=real_random(rng);
        current_arise_e[i]=real_random(rng);
        current_decay_i[i]=real_random(rng);
        current_arise_i[i]=real_random(rng);
        current_decay_e_pre[i]=0;
        current_arise_e_pre[i]=0;
        current_decay_i_pre[i]=0;
        current_arise_i_pre[i]=0;
    }
    double tau_decay=3.5,tau_arise=0.7,tau_lat=0.5;
    double h=0.01;
    double sqrt_h=sqrt(h);
    
    /*Finite-size-noise*/
    double N_e=0.8*N_noise;
    double N_i=0.2*N_noise;
    //** latency delay
    int max_t_lat=int(tau_lat/h);
    double **r_e_lat,**r_i_lat;
    r_e_lat=(double**)malloc(NN*sizeof(double*));
    r_i_lat=(double**)malloc(NN*sizeof(double*));
    for (int i=0;i<NN;i++)
    {
        r_e_lat[i]=(double*)malloc((max_t_lat+1)*sizeof(double));
        r_i_lat[i]=(double*)malloc((max_t_lat+1)*sizeof(double));
        for (int j=0;j<(max_t_tau+1);j++)
        {
            r_e_lat[i][j]=0.0;
            r_i_lat[i][j]=0.0;
        }
    }
    int delay_step=max_t_tau+1;
    
    /*Delay*/
    double **current_decay_e_tem;
    current_decay_e_tem=(double**)malloc(NN*sizeof(double*));
    for (int i=0;i<NN;i++)
    {
        current_decay_e_tem[i]=(double*)malloc((max_t_tau+1)*sizeof(double));
        for (int j=0;j<(max_t_tau+1);j++)
        {
            current_decay_e_tem[i][j]=0.0;
        }
    }
    
    /*Simulation*/
    for(int step=0;step<max_step+1;step++)
    {
 
        //**update current,frequence,tau,convey delay;
        for(int i=0;i<NN;i++)
        {
            i_e_pre[i]=i_e[i];
            i_i_pre[i]=i_i[i];
            eta_ind_pre[i]=eta_ind[i];
            current_arise_e_pre[i]=current_arise_e[i];
            current_decay_e_pre[i]=current_decay_e[i];
            current_arise_i_pre[i]=current_arise_i[i];
            current_decay_i_pre[i]=current_decay_i[i];
            
            r_e[i]=  function_chose_frequency(N_data, i_e_pre[i],origin_data);
            if(N_noise>0)
            {
                std::poisson_distribution<int> distribution_r_e(N_e*r_e[i]*h/1000);
                r_e[i]=distribution_r_e(rng)/(N_e*h/1000);
            }
            tau_e[i]=  function_chose_tau(N_data, i_e_pre[i],origin_data);
            
            r_i[i]=  function_chose_frequency(N_data, i_i_pre[i],origin_data);
            if(N_noise>0)
            {
                std::poisson_distribution<int> distribution_r_i(N_i*r_i[i]*h/1000);
                r_i[i]=distribution_r_i(rng)/(N_i*h/1000);
            }
            tau_i[i]=  function_chose_tau(N_data, i_i_pre[i],origin_data);
            
            r_e_lat[i][0]=r_e[i];
            r_i_lat[i][0]=r_i[i];
        }
        
       
        //**calculate external current;
        for(int i=0;i<NN;i++)
        {
            
            xi_all[i]= normal_distribution(rng);
            xi_ind[i]= normal_distribution(rng);
            eta_ind[i]=eta_ind_pre[i]+h*function_eta(eta_ind_pre[i],tau_ext,xi_ind[i]/sqrt_h,xi_all[0]/sqrt_h, eta_c );
            i_e_ext[i]= function_i_e_ext(i_e_ext0, sigma_ext_e,eta_ind[i]);
            i_i_ext[i]= function_i_i_ext(i_i_ext0, sigma_ext_i,eta_ind[i]);
        }
        
        for(int i=0;i<NN;i++)
        {
            if( step>(max_t_lat ))
            {
                current_arise_e[i]=current_arise_e_pre[i]+h*function_current_arise(current_arise_e_pre[i],  tau_arise,r_e_lat[i][max_t_lat]   );
                current_decay_e[i]=current_decay_e_pre[i]+h*function_current_decay(current_decay_e_pre[i],  tau_decay,current_arise_e[i]  );
                current_arise_i[i]=current_arise_i_pre[i]+h*function_current_arise(current_arise_i_pre[i],  tau_arise,r_i_lat[i][max_t_lat]  );
                current_decay_i[i]=current_decay_i_pre[i]+h*function_current_decay(current_decay_i_pre[i],  tau_decay, current_arise_i[i] );
            }
            else
            {
                current_arise_e[i]=current_arise_e_pre[i]+h*function_current_arise(current_arise_e_pre[i],  tau_arise,r_e[i]   );
                current_decay_e[i]=current_decay_e_pre[i]+h*function_current_decay(current_decay_e_pre[i],  tau_decay,current_arise_e[i]  );
                current_arise_i[i]=current_arise_i_pre[i]+h*function_current_arise(current_arise_i_pre[i],  tau_arise, r_i[i] );
                current_decay_i[i]=current_decay_i_pre[i]+h*function_current_decay(current_decay_i_pre[i],  tau_decay, current_arise_i[i] );
            }
            current_decay_e_tem[i][0]= current_decay_e[i];
        }
        
        
        
        //**calculate current;
        if(step>delay_step)
        {
            for(int i=0;i<NN;i++)
            {
                if(color[i]<1)
                {
                    i_e[i]=i_e_pre[i]+h*delay_function_i_e(i, NN, i_e_pre[i], tau_e[i], i_e_ext[i], omega_ee, omega_ei, current_decay_e, current_decay_i[i],weight,current_decay_e_tem,t_tau);
                    i_i[i]=i_i_pre[i]+h*delay_function_i_i(i, NN, i_i_pre[i], tau_i[i], i_i_ext[i], omega_ie,omega_ii, current_decay_e, current_decay_i[i], weight,current_decay_e_tem,t_tau);
                }
                else
                {
                    i_e[i]= i_e_ext0;
                    i_i[i]= i_i_ext0;
                }
            }
        }
        else
        {
            for(int i=0;i<NN;i++)
            {
                if(color[i]<1)
                {
                    i_e[i]=i_e_pre[i]+h*function_i_e(i, NN, i_e_pre[i], tau_e[i], i_e_ext[i], omega_ee, omega_ei, current_decay_e, current_decay_i[i],weight);
                    i_i[i]=i_i_pre[i]+h*function_i_i(i, NN, i_i_pre[i], tau_i[i], i_i_ext[i], omega_ie, omega_ii, current_decay_e, current_decay_i[i], weight);
                }
                else
                {
                    i_e[i]= i_e_ext0;
                    i_i[i]= i_i_ext0;
                }
            }
        }
        
        //**convey the latency delay data
        
        for (int i = 0; i < NN; i++)
        {
            for (int j = max_t_lat; j > 0; j--)//** be careful og the j range
            {
                r_e_lat[i][j] = r_e_lat[i][j - 1];
                r_i_lat[i][j] = r_i_lat[i][j - 1];
            }
            
        }
        
        //** convey the delay data
        for (int i = 0; i < NN; i++)
        {
            for (int j = max_t_tau; j > 0; j--)//** be careful og the j range
            {
                current_decay_e_tem[i][j] = current_decay_e_tem[i][j - 1];
                
            }
            
        }
        
//        if(step==500 )
//        {
//            for(int i=0;i<NN;i++)
//            {
//                //        i_e[i]=-6.7+0.2/NN*i;
//                i_e[i]=-0;
//                //printf("%d,%f\n",i,i_e[i]);
//                i_i[i]=-1;
//            }
//        }
        
        //**save data
        if(step>500&step%100==0)
        {
            for(int i=0;i<NN;i++)
            {
                if(color[i]<1)
                {
                    fprintf(fp,"%f,",i_e[i]);
                }
            }
            fprintf(fp," \n" );
        }
//        if(step%10000==0)
//                {
//                    printf("%d\n",step);
//                }
    }
    
    fclose(fp);
    return 0;
    
}
 
