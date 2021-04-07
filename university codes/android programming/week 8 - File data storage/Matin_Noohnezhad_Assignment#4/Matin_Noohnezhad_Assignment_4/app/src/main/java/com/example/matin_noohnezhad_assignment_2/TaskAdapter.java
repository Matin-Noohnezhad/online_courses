package com.example.matin_noohnezhad_assignment_2;

import android.content.Context;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;

import java.util.List;

public class TaskAdapter extends RecyclerView.Adapter<TaskAdapter.ViewHolder> {

    private Context context;
    private List<Task> tasks;
    private LayoutInflater layoutInflater;
    private TaskClickListener taskClickListener;
    private boolean isDueTask;

    public TaskAdapter(Context context, List<Task> tasks, TaskClickListener taskClickListener, boolean isDueTask) {
        this.context = context;
        this.tasks = tasks;
        this.layoutInflater = LayoutInflater.from(context);
        this.taskClickListener = taskClickListener;
        this.isDueTask = isDueTask;
    }

    @NonNull
    @Override
    public ViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = layoutInflater.inflate(R.layout.item_task, parent, false);
        return new ViewHolder(view);
    }

    @Override
    public void onBindViewHolder(@NonNull ViewHolder holder, int position) {
        holder.setData(position);
    }

    @Override
    public int getItemCount() {
        return tasks.size();
    }

    public Task removeItem(int position) {
        Task t = tasks.remove(position);
        notifyItemRemoved(position);
        notifyItemRangeChanged(position, getItemCount());
        return t;
    }


    public class ViewHolder extends RecyclerView.ViewHolder implements View.OnClickListener {
        private int position;
        private Task task;
        private ImageView iconIv;
        private TextView taskTitleTv;
        private ImageView deleteTaskIv;
        private TextView taskDescTv;
        private TextView taskDateTv;
        private TextView taskHourTv;
        private ImageView changeTaskStatusIv;


        public ViewHolder(@NonNull View itemView) {
            super(itemView);
            iconIv = itemView.findViewById(R.id.icon_iv);
            taskTitleTv = itemView.findViewById(R.id.task_title_tv);
            deleteTaskIv = itemView.findViewById(R.id.delete_task_iv);
            taskDescTv = itemView.findViewById(R.id.task_desc_tv);
            taskDateTv = itemView.findViewById(R.id.task_date_tv);
            taskHourTv = itemView.findViewById(R.id.task_hour_tv);
            changeTaskStatusIv = itemView.findViewById(R.id.go_to_archive_iv);
            deleteTaskIv.setOnClickListener(this);
            changeTaskStatusIv.setOnClickListener(this);

        }

        public void setData(int position) {
            this.position = position;
            this.task = tasks.get(position);
            taskTitleTv.setText(task.getTitle());
            taskDescTv.setText(task.getDescription());
            //
            taskDateTv.setText((task.getDueDate().getYear()+1900) + "/" + (task.getDueDate().getMonth()+1) + "/" + task.getDueDate().getDate());
            taskHourTv.setText(task.getDueHour().toString() + ":" + task.getDueMinute().toString());
            iconIv.setImageResource(TaskType.findIconByTaskType(task.getType()));
            if (!isDueTask)
                changeTaskStatusIv.setImageResource(R.drawable.go_to_due);
        }

        @Override
        public void onClick(View v) {
            switch (v.getId()) {
                case R.id.delete_task_iv:
//                    members.remove(pos);
//                    notifyDataSetChanged();
                    taskClickListener.onTaskDelete(this.position);
                    break;
                case R.id.go_to_archive_iv:
                    taskClickListener.onTaskChangeStatus(this.position);
                    break;
            }
        }
    }
}
