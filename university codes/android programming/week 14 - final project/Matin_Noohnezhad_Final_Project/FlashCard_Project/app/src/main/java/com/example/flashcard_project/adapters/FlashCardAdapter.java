package com.example.flashcard_project.adapters;

import android.content.Context;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;

import com.example.flashcard_project.R;
import com.example.flashcard_project.beans.FlashCard;
import com.example.flashcard_project.utils.CardClickListener;

import java.util.List;

public class FlashCardAdapter extends RecyclerView.Adapter<FlashCardAdapter.ViewHolder> {

    private Context context;
    private List<FlashCard> cards;
    private LayoutInflater layoutInflater;
    private CardClickListener cardClickListener;

    public FlashCardAdapter(Context context, List<FlashCard> cards, CardClickListener cardClickListener) {
        this.context = context;
        this.cards = cards;
        this.layoutInflater = LayoutInflater.from(context);
        this.cardClickListener = cardClickListener;
    }

    @NonNull
    @Override
    public FlashCardAdapter.ViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = layoutInflater.inflate(R.layout.card_item, parent, false);
        return new ViewHolder(view);
    }

    @Override
    public void onBindViewHolder(@NonNull FlashCardAdapter.ViewHolder viewHolder, int position) {
        viewHolder.setData(position);
    }

    @Override
    public int getItemCount() {
        return cards.size();
    }

    public void addItem(FlashCard card) {
        cards.add(card);
        notifyItemInserted(cards.size() - 1);
    }

    public class ViewHolder extends RecyclerView.ViewHolder implements View.OnClickListener {

        private FlashCard flashCard;
        private TextView wordTv;
        private ImageView questionIv;

        public ViewHolder(@NonNull View itemView) {
            super(itemView);
            wordTv = itemView.findViewById(R.id.word_tv);
            questionIv = itemView.findViewById(R.id.question_iv);
            questionIv.setOnClickListener(this);

        }

        public void setData(int position) {
            this.flashCard = cards.get(position);
            wordTv.setText(flashCard.getWord());
        }

        @Override
        public void onClick(View v) {
            switch (v.getId()) {
                case R.id.question_iv:
                    cardClickListener.onCardClicked(flashCard);
                    break;
            }

        }


    }
}
