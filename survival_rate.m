%% SURVIVAL RATE NY YEARS - 5 10 15 20
clear; clc
datasource='Connection_breast_cancer';
username='root'; password='maquandoarrivalamorte';

connection=database(datasource,username,password);
%ricorda di chiuderla dopo con close(connection)!!

if isopen(connection)==1
    fprintf("\nConnection has been successfully established")
else
    fprintf("\nError encountered while trying to establish a connection")
    fprintf("%s\n", connection.Message) 
    %stampa il messaggio per cui la connectione non è andata a buon fine
end

%%
query = ['USE project_breast_cancer'];
try
    execute(connection,query);
    fprintf("\nQuery execution properly accomplished.")
catch e
    fprintf("\nQuery execution encountered an error.")
    fprintf("%s\n",e.Message)
end
%%
query1=['select case_id from project_breast_cancer.demographic where vital_status="Alive"'];
query2=["select case_id,year_of_death from project_breast_cancer.demographic where vital_status='Dead' and year_of_death is not null"];
try
    not_alive=fetch(connection,query2);
    for i=1:size(not_alive,1)
        query_not_alive=['select year_of_diagnosis, tumor_stage from project_breast_cancer.diagnoses where case_id = "',not_alive.case_id{i},'"'];
        results_not_alive=fetch(connection,query_not_alive);
        not_alive{i,3}=results_not_alive{1,1};
        not_alive{i,4}=results_not_alive{1,2};
    end
    alive=fetch(connection,query1);
    for i=1:size(alive,1)
        query_alive=['select year_of_diagnosis, tumor_stage from project_breast_cancer.diagnoses where case_id = "',alive.case_id{i},'"'];
        results_alive=fetch(connection,query_alive);
        alive{i,2}=results_alive{1,1};
        alive{i,3}=results_alive{1,2};
        
    end    
catch e
    fprintf("\nEsecuzione della query non riuscita\n");
    fprintf("\n%s", e.message);
end
not_alive.Properties.VariableNames(3:4)={'year_of_diagnosis' 'tumor_stage'};
not_alive=movevars(not_alive,'year_of_death','after','tumor_stage');

alive.Properties.VariableNames(2:3)={'year_of_diagnosis' 'tumor_stage'};
fprintf("\nThe number of dead patients is %d",numel(not_alive{:,1}));

%% 5 anni
stage_i=["stage i","stage ia","stage ib"];
stage_ii=["stage ii","stage iia","stage iib"];
stage_iii=["stage iii","stage iiia","stage iiib","stage iiic"];
stage_iv=["stage iv"];

[stage_i_alive,stage_i_not_alive]=division_by_stage(alive,not_alive,stage_i);
[stage_ii_alive,stage_ii_not_alive]=division_by_stage(alive,not_alive,stage_ii);
[stage_iii_alive,stage_iii_not_alive]=division_by_stage(alive,not_alive,stage_iii);
[stage_iv_alive,stage_iv_not_alive]=division_by_stage(alive,not_alive,stage_iv);

alive_i=alive(stage_i_alive,:); not_alive_i=not_alive(stage_i_not_alive,:);
alive_ii=alive(stage_ii_alive,:); not_alive_ii=not_alive(stage_ii_not_alive,:);
alive_iii=alive(stage_iii_alive,:); not_alive_iii=not_alive(stage_iii_not_alive,:);
alive_iv=alive(stage_iv_alive,:); not_alive_iv=not_alive(stage_iv_not_alive,:);
%%
clearvars -except alive not_alive alive_i alive_ii alive_iii alive_iv not_alive_i not_alive_ii not_alive_iii not_alive_iv

%2013 anno della raccolta dati, quindi lo prendiamo come base
sorted_alive= sortrows(alive,'year_of_diagnosis','descend');
for i=1:size(sorted_alive,1)
    if not(isnan(sorted_alive.year_of_diagnosis(i)))
        year_collected_data=sorted_alive.year_of_diagnosis(i);
        break
    end
end
years=[];
for i=1:15
    years(end+1)=i;
end

not_alive_stages={not_alive_i,not_alive_ii,not_alive_iii,not_alive_iv};
alive_stages={alive_i,alive_ii,alive_iii,alive_iv};


count_alive=zeros(size(years,2),size(alive_stages,2)); %anni sulle righe e stadi sulle colonne
count_not_alive=zeros(size(years,2),size(alive_stages,2));
for i=1:length(alive_stages)
    [count_a,count_not_a]=counting_by_years(years,year_collected_data,not_alive_stages{i},alive_stages{i});
    count_not_alive(:,i)=count_not_a;
    count_alive(:,i)=count_a; 
end
%%
matrix_ratio=count_not_alive./count_alive;
matrix_ratio(9:15,:)=[];
years(9:15)=[];
%%
matrix_ratio_new=matrix_ratio;
full=[2,2,2,2];
for j=1:size(matrix_ratio,2)
    for i=1:size(matrix_ratio,1)-1
        matrix_ratio_new(i+1,j)=matrix_ratio_new(i+1,j)+matrix_ratio_new(i,j);
    end
end

matrix_ratio_new=[full;matrix_ratio_new];
years=[0,years];

for i=1:size(matrix_ratio,2)
    plot(years,abs(1-matrix_ratio_new(:,i))*100)
    hold on
end
hold off


%% -------------------UTILITY FUNCTION-------------------------------------
function [stage_alive,stage_not_alive]=division_by_stage(table_alive,table_not_alive,vector_stage)
stage_not_valid=["stage x","not reported"];
stage_alive=zeros(size(table_alive,1),1);
stage_not_alive=zeros(size(table_not_alive,1),1);
for i=1:size(table_alive,1)
    if sum(table_alive.tumor_stage{i}==stage_not_valid)==1
        stage_alive(i)=0;
    elseif sum(table_alive.tumor_stage{i}==vector_stage)==1
        stage_alive(i)=1;
    else
        stage_alive(i)=0;
    end
end
stage_alive=logical(stage_alive);
for j=1:size(table_not_alive,1)
    if sum(table_not_alive.tumor_stage{j}==stage_not_valid)==1
        stage_not_alive(j)=0;
    elseif sum(table_not_alive.tumor_stage{j}==vector_stage)==1
        stage_not_alive(j)=1;
    else
        stage_not_alive(j)=0;
    end
end
stage_not_alive=logical(stage_not_alive);
end

%FUNCTION2
function [count_alive,count_not_alive]=counting_by_years(years,year_data,not_alive,alive)
count_alive=zeros(size(years,2),1); %anni sulle righe e stadi sulle colonne
count_not_alive=zeros(size(years,2),1);
for i=1:size(years,2) %4
    for j=1:size(not_alive,1)%13 etc
        if i==1
            if abs(not_alive.year_of_diagnosis(j)-not_alive.year_of_death(j))<years(i)
                count_not_alive(i)=count_not_alive(i)+1;
            end
        else %i>1
            if abs(not_alive.year_of_diagnosis(j)-not_alive.year_of_death(j))>=years(i-1) && abs(not_alive.year_of_diagnosis(j)-not_alive.year_of_death(j))<years(i)
                count_not_alive(i)=count_not_alive(i)+1;
            end
        end
    end
    for j=1:size(alive,1)
        if i==1
            if (year_data-alive.year_of_diagnosis(j))<=years(i)
            count_alive(i)=count_alive(i)+1;
            end
        else
            if (year_data-alive.year_of_diagnosis(j))>years(i-1) && (year_data-alive.year_of_diagnosis(j))<=years(i)
              count_alive(i)=count_alive(i)+1;
            end
        end
    end
end
            
end