<?php
echo '<!DOCTYPE html>
<html lang="fa" dir="rtl">
<head>
    <meta charset="UTF-8">
    <title>صفحه ادمین</title>
    <link rel="stylesheet" href="../css/admin.css">
</head>
<body>
<div id="d1">
    <div id="d2">
        <div id="d3">
            <div id="d4">
                <button onclick="go_to_category_page()">اضافه کردن دسته بندی</button>
                <button onclick="go_to_category_update()">اصلاح نام دسته بندی</button>
                <button onclick="go_to_update_good()">اصلاح محصولات</button>
                <button onclick="go_to_confirm_page()" id="b2">تایید محصولات اضافه شده</button>
            </div>
        </div>
    </div>
</div>
<script src="../js/admin.js"></script>
</body>
</html>';

?>
