function formPost(categoryId) {


    sendingInput = '<input name="good_id" value=' + categoryId + ' style="display: none">';
    var form = document.getElementById("goodByCategoryForm");
    form.innerHTML = sendingInput;
    form.submit();

}